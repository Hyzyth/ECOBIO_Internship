#!/usr/bin/env python3
"""Verify the shared runtime; --require-cuda also executes GPU forward/backward."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(require_cuda=False, prepare_models=False):
    import torch
    import torchvision
    import sleap_io
    from importlib.metadata import version
    from backend.services.keras_adapter import keras_runtime, KerasAdapter, MODELS
    from backend.services.suite_models import torch_setup
    from tools.sleap_pose import network

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "sleap_nn": version("sleap-nn"),
        "keras_backend": keras_runtime().backend.backend(),
        "cuda_available": torch.cuda.is_available(),
    }
    if require_cuda and not report["cuda_available"]:
        raise RuntimeError(
            "CUDA is unavailable. Check nvidia-smi, the NVIDIA driver and this environment's Torch install. Blackwell needs CUDA 12.8 or newer; use the pinned requirements.txt on Linux x86_64."
        )
    device = torch_setup(device="cuda" if require_cuda else "cpu")
    if require_cuda:
        capability = torch.cuda.get_device_capability()
        arches = torch.cuda.get_arch_list()
        report.update(
            gpu=torch.cuda.get_device_name(),
            capability=list(capability),
            compiled_architectures=arches,
            memory_gib=round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 2
            ),
        )
        architecture = f"sm_{capability[0]}{capability[1]}"
        if capability[0] >= 10 and architecture not in arches:
            raise RuntimeError(
                f"This Torch wheel does not include {architecture}. Reinstall the pinned Blackwell-capable requirements."
            )
    # Actual convolution and optical-flow grid sampling, not just device detection.
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    grid = torch.rand(2, 32, 32, 2, device=device) * 2 - 1
    out = torch.nn.functional.grid_sample(conv(x), grid, align_corners=False)
    out.square().mean().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("Runtime forward/backward produced invalid gradients.")
    pose = network({"filters": 8, "size": 32, "nodes": ["head", "tail"]}).to(device)
    pose(x[:1]).square().mean().backward()
    if require_cuda:
        torch.cuda.synchronize()
    report["forward_backward"] = str(device)
    for name in ["model1.keras", "model2.keras"]:
        adapter = KerasAdapter(MODELS / name)
        model = adapter.load()
        if model.output_shape[-1] != 2:
            raise RuntimeError("Original CNN output contract differs.")
    report["original_cnns_loaded"] = 2
    if prepare_models:
        from backend.services.upstream_architectures import (
            architectures,
            cache_directory,
        )

        motion, temporal = architectures()
        motion(num_images=3).to(device)(torch.zeros(1, 9, 32, 32, device=device))
        report["upstream_architectures"] = str(cache_directory())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--prepare-models", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.require_cuda, args.prepare_models), indent=2))
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
