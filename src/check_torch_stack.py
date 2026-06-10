"""Import torchvision before transformers to surface version mismatches early."""

import re

import torch

# PyTorch ↔ torchvision minor-version pairs (see pytorch.org/get-started/previous-versions)
_TORCHVISION_FOR = {
    "2.8": "0.23",
    "2.7": "0.22",
    "2.6": "0.21",
    "2.5": "0.20",
    "2.4": "0.19",
    "2.3": "0.18",
}


def _torch_minor() -> str:
    return ".".join(torch.__version__.split("+")[0].split(".")[:2])


def _pip_cuda_index() -> str:
    version = torch.__version__
    match = re.search(r"\+cu(\d+)", version)
    if match:
        return f"cu{match.group(1)}"
    if torch.cuda.is_available():
        return "cu126"
    return "cpu"


def _fix_command() -> str:
    torch_mm = _torch_minor()
    tv_mm = _TORCHVISION_FOR.get(torch_mm, "0.23")
    ta_mm = torch_mm
    index = _pip_cuda_index()
    return (
        f"pip install torchvision=={tv_mm}.0 torchaudio=={ta_mm}.0 "
        f"--index-url https://download.pytorch.org/whl/{index}"
    )


def ensure_torchvision_compatible() -> None:
    try:
        import torchvision  # noqa: F401
    except RuntimeError as exc:
        torch_mm = _torch_minor()
        expected = _TORCHVISION_FOR.get(torch_mm, "matching")
        raise RuntimeError(
            f"Incompatible torch ({torch.__version__}) / torchvision: {exc}\n"
            f"torch {torch_mm}.x needs torchvision {expected}.x.\n"
            "If torch was installed with pip (e.g. 2.8.0+cu128), use pip for the "
            "matching domain libraries — do not mix with micromamba pytorch packages:\n"
            f"  {_fix_command()}\n"
            "If an old conda torchvision is present, remove it first:\n"
            "  micromamba remove torchvision torchaudio && " + _fix_command()
        ) from exc
