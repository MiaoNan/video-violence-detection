"""Create isolated directories for generated run artifacts."""
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def create_run_dir(name):
    """Create a unique directory even for simultaneous runs with the same name."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._-") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return Path(tempfile.mkdtemp(prefix=f"{timestamp}_{name}_", dir=OUTPUT_ROOT))


def artifact_path(run_dir, relative_path):
    """Keep a configurable artifact destination inside its run directory."""
    root = Path(run_dir).resolve()
    destination = root / relative_path
    if Path(relative_path).is_absolute() or not destination.resolve().is_relative_to(root):
        raise ValueError("Artifact paths must be relative and stay inside the run directory")
    return destination
