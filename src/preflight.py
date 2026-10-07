"""Safe, shareable runtime diagnostics; never print credentials or environment dumps."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path


def native_bf16_supported(torch_module, device=None) -> bool:
    """Require NVIDIA Ampere+ hardware, not PyTorch's BF16 emulation fallback.

    PyTorch 2.3+ exposes the capability API. In newer releases the default
    is_bf16_supported() can also report emulation on a T4 (compute capability
    7.5); those devices should use native FP16 for this training pilot.
    """
    return bool(torch_module.cuda.is_available()
                and torch_module.cuda.get_device_capability(device)[0] >= 8
                and torch_module.cuda.is_bf16_supported())


def read_fmp_key() -> str | None:
    """Read one named secret without persisting it or revealing exception details."""
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata
        return userdata.get("FMP_API_KEY") or None
    except Exception:
        return None


def runtime_report(*, check_secret: bool = False, output_path=None) -> dict:
    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "in_colab": bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").exists(),
        "dependencies": {},
        "checks": [],
    }
    for package in ("torch", "numpy", "pandas", "scipy", "requests", "pytest"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = None
    root = Path("/content") if Path("/content").exists() else Path.cwd()
    report["local_disk_free_gb"] = round(shutil.disk_usage(root).free / 2**30, 2)
    try:
        report["system_ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    except (ValueError, OSError):
        report["system_ram_gb"] = None
    report["drive_mounted"] = Path("/content/drive/MyDrive").is_dir()
    report["fmp_secret"] = "not_checked"
    if check_secret:
        report["fmp_secret"] = "available" if read_fmp_key() else "missing_or_not_authorized"
    try:
        import torch
        torch_major_minor = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
        report["torch_api_compatible"] = torch_major_minor >= (2, 3)
        if not report["torch_api_compatible"]:
            report["checks"].append("This pilot requires PyTorch 2.3 or newer for the mixed-precision API.")
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = torch.version.cuda
        if report["cuda_available"]:
            prop = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": prop.name, "memory_gb": round(prop.total_memory / 2**30, 2)}
            report["native_bf16_supported"] = native_bf16_supported(torch, 0)
            report["recommended_precision"] = "bf16" if report["native_bf16_supported"] else "fp16"
            x = torch.randn(128, 128, device="cuda", requires_grad=True)
            (x @ x.T).square().mean().backward()
            torch.cuda.synchronize()
            report["gpu_forward_backward"] = bool(torch.isfinite(x.grad).all().item())
            del x
        else:
            report["gpu"] = None
            report["recommended_precision"] = "fp32"
            report["gpu_forward_backward"] = False
            report["checks"].append("Select Runtime > Change runtime type > a GPU before training.")
    except Exception as exc:
        report["cuda_available"] = False
        report["gpu_forward_backward"] = False
        report["checks"].append("PyTorch device check failed: " + type(exc).__name__)
    missing = [k for k,v in report["dependencies"].items() if v is None and k != "pytest"]
    if missing:
        report["checks"].append("Missing dependencies: " + ", ".join(missing))
    if report["local_disk_free_gb"] < 5:
        report["checks"].append("Fewer than 5 GB available on local runtime disk.")
    report["ready_for_gpu_smoke_test"] = bool(report.get("gpu_forward_backward") and report.get("torch_api_compatible") and not missing)
    report["live_data_training_verified"] = False
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_checkpoint_directory(directory) -> dict:
    """Test only the explicitly selected output folder; no file listings are collected."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="fjepa_write_check_", dir=directory) as handle:
        handle.write(b"F-JEPA checkpoint destination check\n")
        handle.flush()
    return {"checkpoint_directory_writable": True,
            "persistent_drive_destination": str(directory).startswith("/content/drive/MyDrive/")}


if __name__ == "__main__":
    print(json.dumps(runtime_report(output_path="setup_report.json"), indent=2))
