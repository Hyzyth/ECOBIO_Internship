"""CCRT models sharing the app's PyTorch runtime.

A: torchvision ResNet-18. B: SLEAP confidence maps + sklearn random forest.
C: DeepEthogram TinyMotionNet, hidden two-stream ResNet features and TGM.
All model families share RGB well crops and the same coma/awake/empty targets.
"""

from collections import OrderedDict, deque
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
import numpy as np
import cv2
from flyscope.models.keras import crop_well, occupancy_features
from flyscope.imaging.foreground import MASKED, ISOLATED, PREPROCESSINGS, isolate

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CLASSES = ["coma", "awake", "empty"]
PREPROCESSING = MASKED
POSE_PREPROCESSING = "foreground-centered-square-rgb-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def crop_rgb(image, points, size=128, preprocessing=PREPROCESSING):
    if preprocessing not in PREPROCESSINGS:
        raise ValueError("Unsupported crop preprocessing.")
    crop, mask = crop_well(image, points)
    features, _ = occupancy_features(crop, mask)
    if preprocessing == ISOLATED:
        crop, _ = isolate(crop, mask)
    crop[mask == 0] = 255
    side = max(crop.shape[:2])
    square = np.full((side, side, 3), 255, np.uint8)
    left, top = (side - crop.shape[1]) // 2, (side - crop.shape[0]) // 2
    square[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
    return cv2.cvtColor(cv2.resize(square, (size, size)), cv2.COLOR_BGR2RGB), features


def torch_setup(seed=2026, device="auto"):
    import torch

    torch.set_num_threads(int(os.environ.get("FLYSCOPE_TORCH_THREADS", "2")))
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested, but this runtime has no available CUDA device."
        )
    # CUDA grid_sample backward (optical flow) has no deterministic kernel.
    # Preserve seeded selections/training; warn about that kernel instead of
    # aborting all GPU flow training. CPU keeps strict deterministic checks.
    torch.use_deterministic_algorithms(True, warn_only=device == "cuda")
    return torch.device(device)


def tensor(images, device):
    import torch

    return (
        torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy())
        .float()
        .to(device)
        / 255
    )


def normalize(x):
    import torch

    return (
        x - torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]


def make_resnet(initialization="imagenet"):
    from torchvision.models import resnet18, ResNet18_Weights
    import torch

    net = resnet18(
        weights=ResNet18_Weights.DEFAULT if initialization == "imagenet" else None
    )
    net.fc = torch.nn.Linear(net.fc.in_features, 3)
    return net


def make_deepethogram(window=5, initialization="imagenet", feature_mode="two-stream"):
    import torch
    from torchvision.models import resnet18
    from flyscope.models.upstream import architectures

    TinyMotionNet, TGM = architectures()

    class Fusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1024, 3)

        def forward(self, rgb, motion):
            return self.fc(torch.cat([rgb, motion], dim=1))

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dimension = 512 if feature_mode == "appearance" else 1024
            self.motion = TinyMotionNet(num_images=window)

            def initialized_cnn(channels):
                backbone = resnet18(weights=None)
                if channels != 3:
                    backbone.conv1 = torch.nn.Conv2d(channels, 64, 7, 2, 3, bias=False)
                backbone.fc = torch.nn.Identity()
                # Match the original DeepEthogram feature/classifier key layout.
                cnn = torch.nn.Sequential(backbone, torch.nn.Identity())
                if initialization == "imagenet":
                    from torchvision.models import ResNet18_Weights

                    state = ResNet18_Weights.DEFAULT.get_state_dict(
                        progress=True, check_hash=True
                    )
                    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
                    if channels != 3:
                        state["conv1.weight"] = (
                            state["conv1.weight"]
                            .mean(dim=1, keepdim=True)
                            .repeat(1, channels, 1, 1)
                        )
                    cnn[0].load_state_dict(state, strict=True)
                return cnn

            spatial = initialized_cnn(3)
            flow = initialized_cnn((window - 1) * 2)
            self.spatial, self.flow, self.fusion = spatial, flow, Fusion()
            if feature_mode == "appearance":
                self.appearance_head = torch.nn.Linear(512, 3)
            self.temporal = TGM(
                D=self.feature_dimension,
                classes=3,
                filter_length=window,
                n_filters=3,
                num_layers=2,
                c_out=2,
                input_dropout=0,
                dropout_p=0.2,
            )

        def features(self, clips):
            b, t, c, h, w = clips.shape
            if feature_mode == "appearance":
                rgb = self.spatial(normalize(clips[:, -1]))
                return rgb, self.appearance_head(rgb)
            with torch.no_grad():
                flows = self.motion(clips.reshape(b, t * c, h, w))[0]
            rgb = self.spatial(normalize(clips[:, -1]))
            motion = self.flow(flows)
            return torch.cat([rgb, motion], dim=1), self.fusion(rgb, motion)

    return Network()


def frozen_batch_norm(network):
    # Tiny subsets can have batch size 1; keep ImageNet BN statistics rather
    # than fitting noisy moments or failing on a 1x1 feature map.
    import torch

    for m in network.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def flow_loss(network, clips):
    """Photometric + smoothness objective, only on selected training wells."""
    import torch
    import torch.nn.functional as F

    b, t, c, h, w = clips.shape
    flows = network.motion(clips.reshape(b, t * c, h, w))[0]
    fh, fw = flows.shape[-2:]
    flows = F.interpolate(flows, size=(h, w), mode="bilinear", align_corners=False)
    flows[:, 0::2] = flows[:, 0::2] * w / fw
    flows[:, 1::2] = flows[:, 1::2] * h / fh
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=clips.device),
        torch.linspace(-1, 1, w, device=clips.device),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=-1)[None].expand(b, -1, -1, -1)
    loss = 0
    for i in range(t - 1):
        f = flows[:, 2 * i : 2 * i + 2]
        delta = torch.stack(
            [f[:, 0] * 2 / max(w - 1, 1), f[:, 1] * 2 / max(h - 1, 1)], dim=-1
        )
        warped = F.grid_sample(
            clips[:, i],
            base + delta,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        photometric = torch.sqrt((warped - clips[:, i + 1]).square() + 1e-6).mean()
        smooth = (f[:, :, 1:] - f[:, :, :-1]).abs().mean() + (
            f[:, :, :, 1:] - f[:, :, :, :-1]
        ).abs().mean()
        loss = loss + photometric + 0.01 * smooth
    return loss / (t - 1)


def pose_features(pose, foreground):
    points = pose["points"]
    xy = np.array([[p["x"], p["y"]] for p in points])
    visible = np.array([p["visible"] for p in points], bool)
    distances = []
    for a, b in itertools.combinations(range(len(points)), 2):
        distances.append(
            float(np.linalg.norm(xy[a] - xy[b])) if visible[a] and visible[b] else -1
        )
    valid = [x for x in distances if x > 0]
    scale = np.median(valid) if valid else 1
    return np.array(
        [
            visible.mean(),
            np.mean([p["score"] for p in points]),
            *[d / scale if d >= 0 else -1 for d in distances],
        ],
        np.float32,
    )


def pose_input(rgb):
    """Match centered-instance pose pretraining without using behavior labels."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    interior = cv2.erode((gray < 250).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if interior.sum() < 25:
        return rgb
    median = np.median(gray[interior])
    mad = np.median(np.abs(gray[interior] - median))
    animal = interior & (
        (gray < median - max(25, 4 * mad)) | ((hsv[..., 1] > 65) & (hsv[..., 2] < 230))
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(animal.astype(np.uint8))
    if n < 2:
        return rgb
    x, y, w, h, area = stats[1 + int(stats[1:, 4].argmax())]
    if area < 4:
        return rgb
    side = max(12, int(max(w, h) * 2.2))
    left = int(x + w / 2 - side / 2)
    top = int(y + h / 2 - side / 2)
    crop = np.full((side, side, 3), 255, np.uint8)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(rgb.shape[1], left + side), min(rgb.shape[0], top + side)
    crop[y0 - top : y1 - top, x0 - left : x1 - left] = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (rgb.shape[1], rgb.shape[0]))


def model_version(meta):
    keys = [
        "architecture",
        "image_size",
        "window",
        "preprocessing",
        "pose_preprocessing",
        "classes",
        "recovery_policy",
        "artifact_hashes",
        "selection",
        "training_dataset_sha256",
        "class_support",
        "initialization",
        "pose_parent_sha256",
    ]
    if "pose_quality_policy" in meta:
        keys.append("pose_quality_policy")
    if "training_sampling" in meta:
        keys.append("training_sampling")
    for key in ("deepethogram_variant", "appearance_parent_version"):
        if key in meta:
            keys.append(key)
    return hashlib.sha256(
        json.dumps({k: meta.get(k) for k in keys}, sort_keys=True).encode()
    ).hexdigest()


POSE_QUALITY_POLICY = {
    "minimum_visible_keypoints": 2,
    "minimum_spread_fraction": 0.05,
    "note": "Reject absent/collapsed poses; this is a screening rule, not proof of anatomical accuracy.",
}


def usable_pose(pose, policy=POSE_QUALITY_POLICY):
    visible = [p for p in pose["points"] if p["visible"]]
    spread = (
        np.ptp([[p["x"], p["y"]] for p in visible], axis=0).max()
        if len(visible) > 1
        else 0
    )
    return (
        len(visible) >= policy["minimum_visible_keypoints"]
        and spread >= policy["minimum_spread_fraction"]
    )


class PoseWorker:
    """Shared-environment pose inference; no interpreter configuration required."""

    def __init__(self, model, device="cpu"):
        self.root = Path(model).resolve()
        self.device = device

    def predict(self, images, identities, batch_size=8):
        from flyscope.training.pose import infer_images

        return infer_images(
            self.root, images, identities, device=self.device, batch_size=batch_size
        )

    def close(self):
        from flyscope.training.pose import clear_cache

        clear_cache()


def forest_export(forest):
    return {
        "classes": [int(v) for v in forest.classes_],
        "trees": [
            {
                "left": t.tree_.children_left.tolist(),
                "right": t.tree_.children_right.tolist(),
                "feature": t.tree_.feature.tolist(),
                "threshold": t.tree_.threshold.tolist(),
                "values": t.tree_.value[:, 0, :].tolist(),
            }
            for t in forest.estimators_
        ],
    }


def forest_probabilities(forest, features):
    x = np.asarray(features)
    result = np.zeros((len(x), 3))
    # Traverse all samples together, avoiding the Python sample × tree loop.
    for tree in forest["trees"]:
        left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
        columns, thresholds = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
        nodes = np.zeros(len(x), dtype=int)
        active = np.flatnonzero(left[nodes] >= 0)
        while len(active):
            current = nodes[active]
            nodes[active] = np.where(
                x[active, columns[current]] <= thresholds[current],
                left[current],
                right[current],
            )
            active = active[left[nodes[active]] >= 0]
        values = np.asarray(tree["values"])[nodes]
        result[:, forest["classes"]] += values / values.sum(axis=1, keepdims=True)
    return result / len(forest["trees"])


def compile_forest(forest):
    """Convert portable lists once per worker, rather than once per frame."""
    if forest is None:
        return None
    return {
        "classes": forest["classes"],
        "trees": [
            {key: np.asarray(value) for key, value in tree.items()}
            for tree in forest["trees"]
        ],
    }


class SuiteModel:
    def __init__(self, manifest, device="auto"):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.meta = json.loads(self.path.read_text())
        if (
            self.meta.get("classes") != CLASSES
            or self.meta.get("preprocessing") not in PREPROCESSINGS
            or self.meta.get("version") != model_version(self.meta)
        ):
            raise ValueError(
                "Model interpretation/version does not match its metadata."
            )
        for name, expected in self.meta["artifact_hashes"].items():
            file = (self.root / name).resolve()
            if not file.is_relative_to(self.root) or digest(file) != expected:
                raise ValueError("Model artifact does not match its metadata hashes.")
        self.kind = self.meta["architecture"]
        self.size = self.meta["image_size"]
        self.sessions = OrderedDict()
        self.pose = None
        if self.kind == "sleap-rf":
            pose_preprocessing = json.loads(
                (self.root / "pose/pose.json").read_text()
            ).get("source_crop_preprocessing")
            if (
                pose_preprocessing is not None
                and pose_preprocessing != self.meta["preprocessing"]
            ):
                raise ValueError(
                    "Pose parent and behavior checkpoint use different crop preprocessing."
                )
            self.forest = {
                name: compile_forest(value)
                for name, value in json.loads(
                    (self.root / "forest.json").read_text()
                ).items()
                if name in ("occupancy", "state")
            }
            self.pose = PoseWorker(self.root / "pose", device=device)
        else:
            import torch

            self.device = torch_setup(device=device)
            self.network = (
                make_resnet("scratch")
                if self.kind == "resnet18"
                else make_deepethogram(
                    self.meta["window"],
                    "scratch",
                    (
                        "appearance"
                        if self.meta.get("deepethogram_variant") == "resnet-temporal"
                        else "two-stream"
                    ),
                )
            )
            self.network.load_state_dict(
                torch.load(
                    self.root / "model.pt", map_location=self.device, weights_only=True
                ),
                strict=True,
            )
            self.network.to(self.device).eval()

    def predict(self, content, wells, session=None, frame=None):
        if not wells:
            return []
        started = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unreadable model image.")
        decoded = time.perf_counter()
        images = []
        foreground = []
        unreliable_pose = [False] * len(wells)
        for well in wells:
            im, feature = crop_rgb(
                image, well["points"], self.size, self.meta["preprocessing"]
            )
            images.append(im)
            foreground.append(feature)
        cropped = time.perf_counter()
        if self.kind == "sleap-rf":
            poses = self.pose.predict(
                [pose_input(im) for im in images], [w["id"] for w in wells]
            )
            quality = self.meta.get("pose_quality_policy")
            if quality:
                unreliable_pose = [not usable_pose(p, quality) for p in poses]
            occupancy = forest_probabilities(self.forest["occupancy"], foreground)
            states = (
                forest_probabilities(
                    self.forest["state"], [pose_features(p, None) for p in poses]
                )
                if self.forest["state"]
                else np.zeros((len(wells), 3))
            )
            probabilities = states * occupancy[:, 0, None]
            probabilities[:, 2] = occupancy[:, 2]
        elif self.kind == "resnet18":
            import torch

            with torch.inference_mode():
                probabilities = (
                    self.network(normalize(tensor(images, self.device)))
                    .softmax(1)
                    .cpu()
                    .numpy()
                )
        else:
            if session is None or frame is None:
                raise ValueError(
                    "DeepEthogram requires an ordered sequence, not an independent image."
                )
            import torch

            state = self.sessions.pop(session, {"last_frame": None, "wells": {}})
            if state["last_frame"] is not None and frame != state["last_frame"] + 1:
                raise ValueError(
                    "Temporal inference frames must be contiguous and ordered. Start a new analysis session."
                )
            window = self.meta["window"]
            clips = []
            features = []
            for well, im in zip(wells, images):
                previous = state["wells"].setdefault(
                    well["id"],
                    {
                        "images": deque(maxlen=window),
                        "features": deque(maxlen=window),
                        "last_frame": None,
                    },
                )
                if (
                    previous["last_frame"] is not None
                    and frame != previous["last_frame"] + 1
                ):
                    previous["images"].clear()
                    previous["features"].clear()
                previous["images"].append(im)
                previous["last_frame"] = frame
                history = list(previous["images"])
                clips.append([history[0]] * (window - len(history)) + history)
            x = tensor([im for clip in clips for im in clip], self.device).reshape(
                len(wells), window, 3, self.size, self.size
            )
            with torch.inference_mode():
                current, fusion_logits = self.network.features(x)
                for w, v in zip(wells, current):
                    past = state["wells"][w["id"]]["features"]
                    past.append(v.detach())
                    history = list(past)
                    features.append(
                        torch.stack(
                            [history[0]] * (window - len(history)) + history, dim=1
                        )
                    )
                logits = (
                    fusion_logits
                    if self.meta.get("deepethogram_variant") == "two-stream"
                    else self.network.temporal(torch.stack(features))[:, :, -1]
                )
                probabilities = logits.softmax(1).cpu().numpy()
            state["last_frame"] = frame
            self.sessions[session] = state
            while len(self.sessions) > 4:
                self.sessions.popitem(last=False)
        inferred = time.perf_counter()
        self.last_timing = {
            "decode_s": decoded - started,
            "crop_s": cropped - decoded,
            "model_s": inferred - cropped,
            "well_count": len(wells),
        }
        records = []
        for well, p, bad_pose in zip(wells, probabilities, unreliable_pose):
            if (
                not np.isfinite(p).all()
                or (p < 0).any()
                or not np.isclose(p.sum(), 1, atol=1e-4)
            ):
                raise RuntimeError("Nonfinite or invalid model probabilities.")
            prediction = CLASSES[int(p.argmax())]
            abstain = bool(well.get("partial")) or (bad_pose and prediction != "empty")
            records.append(
                {
                    "well_id": well["id"],
                    "boundary_partial": bool(well.get("partial", False)),
                    "decision_eligible": not abstain,
                    "abstention_reason": (
                        "partial_boundary"
                        if well.get("partial")
                        else "unusable_pose" if abstain else None
                    ),
                    "prediction": "unknown" if abstain else prediction,
                    "confidence": None if abstain else float(p.max()),
                    "state_probabilities": dict(zip(CLASSES, map(float, p))),
                    "raw_prediction": prediction,
                    "raw_confidence": float(p.max()),
                    "class_support": self.meta["class_support"],
                }
            )
        return records

    def close(self):
        if self.pose:
            self.pose.close()
