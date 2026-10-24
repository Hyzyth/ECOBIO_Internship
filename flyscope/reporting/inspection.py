"""Inspect checkpoint inputs, native resolution, ResNet maps and class Grad-CAM."""

import argparse
import html
import json
from pathlib import Path
import cv2
import numpy as np
from flyscope.data.datasets import Dataset
from flyscope.models.keras import crop_well
from flyscope.models.runtime import SuiteModel, crop_rgb, tensor, normalize, CLASSES


def resnet_maps(model, rgb):
    import torch

    if model.kind != "resnet18":
        raise ValueError("Layer maps and Grad-CAM require a ResNet-18 checkpoint.")
    activations, hooks = {}, []

    def capture(name):
        def hook(module, args, output):
            activations[name] = output

        return hook

    try:
        for name in ("layer1", "layer2", "layer3", "layer4"):
            hooks.append(
                getattr(model.network, name).register_forward_hook(capture(name))
            )
        with torch.enable_grad():
            x = normalize(tensor([rgb], model.device)).requires_grad_(True)
            logits = model.network(x)
            probabilities = logits.softmax(1)[0].detach().cpu().tolist()
            maps = {}
            for index, label in enumerate(CLASSES[:2]):
                gradients = torch.autograd.grad(
                    logits[0, index], activations["layer4"], retain_graph=True
                )[0]
                cam = (
                    (gradients.mean((2, 3), keepdim=True) * activations["layer4"])
                    .sum(1)
                    .relu()[0]
                )
                value = cam.detach().cpu().numpy()
                value = cv2.resize(value, (rgb.shape[1], rgb.shape[0]))
                value /= max(float(value.max()), 1e-12)
                color = cv2.cvtColor(
                    cv2.applyColorMap(
                        (value * 255).astype(np.uint8), cv2.COLORMAP_TURBO
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                maps[f"gradcam-{label}"] = cv2.addWeighted(rgb, 0.55, color, 0.45, 0)
            for name, output in activations.items():
                channels = output[0].detach().cpu().numpy()
                top = np.argsort(np.mean(np.abs(channels), axis=(1, 2)))[-4:][::-1]
                panels = []
                for i in top:
                    value = channels[i]
                    value = (value - value.min()) / max(float(np.ptp(value)), 1e-12)
                    panel = cv2.resize(
                        (value * 255).astype(np.uint8),
                        (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    panels.append(cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB))
                maps[name] = np.concatenate(panels, axis=1)
        return dict(zip(CLASSES, probabilities)), maps
    finally:
        for hook in hooks:
            hook.remove()


def flow_maps(model, images):
    import torch

    if model.meta.get("deepethogram_variant") == "resnet-temporal":
        return {}, {"motion": "disabled: frozen reference ResNet features only"}
    size, window = model.size, model.meta["window"]
    x = tensor(images, model.device).reshape(1, window * 3, size, size)
    panels, magnitudes = {}, []
    with torch.inference_mode():
        flows = model.network.motion(x)[0][0].cpu().numpy()
    fh, fw = flows.shape[-2:]
    for i in range(window - 1):
        horizontal = cv2.resize(flows[2 * i], (size, size)) * size / fw
        vertical = cv2.resize(flows[2 * i + 1], (size, size)) * size / fh
        magnitude, angle = cv2.cartToPolar(horizontal, vertical, angleInDegrees=True)
        hsv = np.zeros((size, size, 3), np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(
            magnitude / max(float(np.percentile(magnitude, 95)), 1e-6) * 255, 0, 255
        ).astype(np.uint8)
        panels[f"flow-{i + 1}-to-{i + 2}"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        panels[f"context-{i + 1}"] = images[i]
        magnitudes.append(
            {
                "pair": [i + 1, i + 2],
                "median_pixels": float(np.median(magnitude)),
                "max_pixels": float(magnitude.max()),
            }
        )
    return panels, {
        "magnitudes": magnitudes,
        "legend": "Hue: direction. Brightness: magnitude, scaled to this pair's 95th percentile. Inspect alongside original causal frames; low reconstruction loss is not evidence of fly motion.",
    }


def inspect(dataset, output, model_path=None, samples=12, device="auto"):
    if samples < 1:
        raise ValueError("Samples must be positive.")
    out = Path(output)
    if out.exists():
        raise ValueError("Use a new inspection output directory.")
    root = Path(dataset)
    role = json.loads((root / "annotations.json").read_text()).get("role", "train")
    data = Dataset(root, role)
    model = SuiteModel(model_path, device) if model_path else None
    out.mkdir(parents=True)
    rows, gallery = [], []
    try:
        candidates = sorted(
            (
                k
                for k in data.geometry
                if k[1] >= data.manifest.get("analysis_start_index", 0)
            ),
            key=lambda k: (k[1], k[0]),
        )
        chosen = (
            np.linspace(
                0, len(candidates) - 1, min(samples, len(candidates)), dtype=int
            )
            if candidates
            else []
        )
        images = {im["frame_index"]: im for im in data.manifest["images"]}
        for i in chosen:
            uid, frame = candidates[i]
            geometry = data.geometry[(uid, frame)]
            image = data.image(images[frame]["path"])
            native, _ = crop_well(image, geometry["points"])
            row = {
                "well_uid": uid,
                "frame_index": frame,
                "source_crop_width": native.shape[1],
                "source_crop_height": native.shape[0],
            }
            side = max(native.shape[:2])
            row["native_square_side"] = side
            row["size_ratios"] = {
                str(size): size / side for size in (128, 200, 224, 256)
            }
            if model:
                rgb, _ = crop_rgb(
                    image, geometry["points"], model.size, model.meta["preprocessing"]
                )
                prefix = f"sample-{len(rows) + 1:03d}"
                panels = {"input": rgb}
                if model.kind == "resnet18":
                    row["probabilities"], maps = resnet_maps(model, rgb)
                    panels.update(maps)
                elif model.kind == "deepethogram":
                    start = data.manifest.get("analysis_start_index", min(images))
                    clip = []
                    for f in range(frame - model.meta["window"] + 1, frame + 1):
                        f = max(start, f)
                        g = data.geometry.get((uid, f))
                        if g is None:
                            raise ValueError(
                                "A causal context image/boundary is missing; inspect the complete analysis window."
                            )
                        clip.append(
                            crop_rgb(
                                data.image(images[f]["path"]),
                                g["points"],
                                model.size,
                                model.meta["preprocessing"],
                            )[0]
                        )
                    maps, row["optical_flow"] = flow_maps(model, clip)
                    panels.update(maps)
                for name, panel in panels.items():
                    filename = f"{prefix}-{name}.png"
                    cv2.imwrite(
                        str(out / filename), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                    )
                    gallery.append(
                        f'<figure><img src="{filename}" alt="{name}"><figcaption>{html.escape(uid)} · frame {frame + 1} · {name}</figcaption></figure>'
                    )
            rows.append(row)
        sides = [r["native_square_side"] for r in rows]
        report = {
            "dataset_sha256": data.sha,
            "model_version": model.meta["version"] if model else None,
            "model_input_size": model.size if model else None,
            "pose_network_size": (
                json.loads((model.root / "pose/pose.json").read_text())[
                    "configuration"
                ]["size"]
                if model and model.kind == "sleap-rf"
                else None
            ),
            "native_side_pixels": (
                {
                    "min": min(sides),
                    "median": float(np.median(sides)),
                    "max": max(sides),
                }
                if sides
                else None
            ),
            "contracts": {
                "original_cnn": "200x200 grayscale, fixed by supplied weights",
                "resnet18": "RGB; checkpoint size is binding; ImageNet reference is 224, not an accuracy guarantee",
                "sleap-rf": "RGB well at checkpoint size, foreground-centered, then resized to pose.json configuration.size",
                "deepethogram": "RGB causal clips; checkpoint size (multiple of 8, at least 32) and window are binding; compare 128/192 and measure throughput",
            },
            "interpretation": "Ratios >1 upscale existing pixels; they add no anatomical detail. Compare 128/224/256 using training/validation only. Grad-CAM is class sensitivity, not proof of anatomical reasoning; layer images show the four strongest channels, independently scaled.",
            "samples": rows,
        }
        (out / "inspection.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        (out / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>FlyScope model inspection</title><style>body{font:16px sans-serif;margin:2em}img{max-width:100%;image-rendering:pixelated}figure{display:inline-block;vertical-align:top;max-width:45%}</style><h1>Inputs and ResNet inspection</h1><p>'
            + html.escape(report["interpretation"])
            + '</p><p><a href="inspection.json">Resolution and probability report</a></p>'
            + "".join(gallery)
        )
        return report
    finally:
        if model:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        help="Suite model.json; ResNet also produces class Grad-CAM and layer maps",
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    result = inspect(args.dataset, args.output, args.model, args.samples, args.device)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
