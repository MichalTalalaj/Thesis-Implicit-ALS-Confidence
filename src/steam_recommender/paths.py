"""Checkout-relative inputs and optionally isolated result paths.

Importing this module never creates directories or loads research data.
"""
import os
from pathlib import Path


def project_root():
    """Resolve this package's checkout, independently of the working directory."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Use this package from its checkout (pip install -e .).")
    return root


def project_path(relative):
    """Resolve a repository path, routing results through STEAM_RESULTS_DIR.

    Relative override values are relative to the checkout, not the kernel's cwd.
    Data inputs and prepared data always remain under data/.
    """
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a relative path inside the project")
    root = project_root()
    if relative.parts and relative.parts[0] == "results":
        results = Path(os.environ.get("STEAM_RESULTS_DIR", "results"))
        if not results.is_absolute():
            results = root / results
        return results.joinpath(*relative.parts[1:])
    return root / relative
