"""Helpers for building the ESP-IDF firmware for one configuration."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import emlearn
from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile

from bench.config import BUILD, FIRMWARE, PROFILES, generated_dir

BUILD_TIMEOUT_S = 3600
TOO_LARGE = re.compile(r"overflow|too small|does not fit|exceeds", re.IGNORECASE)


def idf_command() -> list[str]:
    """Run idf.py with the ESP-IDF Python environment set up by export.sh."""
    idf_path = os.environ.get("IDF_PATH")
    env_path = os.environ.get("IDF_PYTHON_ENV_PATH")
    if idf_path and env_path:
        python = Path(env_path) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return [str(python), str(Path(idf_path) / "tools" / "idf.py")]
    return ["idf.py"]


def build_dir(profile: str) -> Path:
    return BUILD / profile


def idf_args(profile: str) -> list[str]:
    target, _mhz, extra_defaults = PROFILES[profile]
    bdir = build_dir(profile)
    defaults = ";".join(["sdkconfig.defaults", *extra_defaults])
    return [
        *idf_command(), "-C", str(FIRMWARE), "-B", str(bdir),
        f"-DIDF_TARGET={target}",
        f"-DSDKCONFIG={bdir / 'sdkconfig'}",
        f"-DSDKCONFIG_DEFAULTS={defaults}",
        f"-DMODEL_DIR={bdir / 'model'}",
        f"-DEMLEARN_INCLUDE_DIR={os.path.dirname(emlearn.__file__)}",
    ]


def stage(profile: str, dataset: str, model: str, tool: str) -> None:
    """Copy one configuration into build/<profile>/model, the directory the firmware compiles."""
    dst = build_dir(profile) / "model"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in generated_dir(dataset, model, tool).iterdir():
        if f.suffix in (".h", ".c", ".cpp"):
            shutil.copyfile(f, dst / f.name)  # new mtime, so ninja recompiles


def section_sizes(elf_path: Path) -> dict[str, int]:
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        return {s.name: s["sh_size"] for s in elf.iter_sections()
                if s["sh_flags"] & SH_FLAGS.SHF_ALLOC and s["sh_size"] > 0}


def build(profile: str, dataset: str, model: str, tool: str) -> dict:
    """Stage and build one configuration. Returns status and sizes."""
    stage(profile, dataset, model, tool)
    try:
        r = subprocess.run([*idf_args(profile), "build"], capture_output=True, text=True,
                           timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"status": "build_timeout", "detail": f"no result after {BUILD_TIMEOUT_S}s"}
    log = r.stdout + r.stderr
    if r.returncode:
        errors = [line for line in log.splitlines() if "error" in line.lower()]
        detail = (errors[-1] if errors else log.strip().splitlines()[-1])[:300]
        status = "too_large" if TOO_LARGE.search(log) else "build_error"
        return {"status": status, "detail": detail}
    bdir = build_dir(profile)
    sections = section_sizes(bdir / "bench.elf")
    return {"status": "ok", "bin_bytes": (bdir / "bench.bin").stat().st_size, "sections": sections}


def flash(profile: str, port: str) -> None:
    subprocess.run([*idf_args(profile), "-p", port, "flash"], check=True,
                   capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
