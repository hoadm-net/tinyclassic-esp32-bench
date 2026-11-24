"""Helpers for building the ESP-IDF firmware for one configuration."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import emlearn
from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile

from bench import proc
from bench.config import BUILD, FIRMWARE, PROFILES, generated_dir

BUILD_TIMEOUT_S = 3600

# Build failures we classify, highest priority first: (status, pattern matching the key log line).
FAILURES = [
    # A memory region or the app partition is too small: a hard limit.
    ("too_large", re.compile(r"region `[\w.]+' overflowed by \d+ bytes"
                             r"|partition is too small[^\n]*|too small for binary[^\n]*")),
    # Xtensa L32R can only load literals placed up to 256 KB before the instruction.
    ("literal_range", re.compile(r"dangerous relocation: l32r: literal target out of range")),
]


def classify_failure(log: str) -> tuple[str, str]:
    found = [(status, m.group(0)) for status, pattern in FAILURES if (m := pattern.search(log))]
    if found:
        return found[0][0], "; ".join(detail for _, detail in found)[:300]
    errors = [line for line in log.splitlines() if "error" in line.lower()]
    return "build_error", (errors[-1] if errors else log.strip().splitlines()[-1])[:300]


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


def idf_args(profile: str, litpools: bool = False) -> list[str]:
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
        f"-DMODEL_LITPOOLS={'ON' if litpools else 'OFF'}",
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


def build(profile: str, dataset: str, model: str, tool: str, litpools: bool = False) -> dict:
    """Stage and build one configuration. Returns status and sizes."""
    stage(profile, dataset, model, tool)
    try:
        r = proc.run([*idf_args(profile, litpools), "build"], timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"status": "build_timeout", "detail": f"no result after {BUILD_TIMEOUT_S}s"}
    if r.returncode:
        log = r.stdout + r.stderr
        # idf.py writes the full tool output to files and only names them on failure.
        for path in re.findall(r"(\S+/log/idf_py_std(?:out|err)_output_\d+)", log):
            if Path(path).exists():
                log += Path(path).read_text(errors="replace")
        status, detail = classify_failure(log)
        return {"status": status, "detail": detail}
    bdir = build_dir(profile)
    sections = section_sizes(bdir / "bench.elf")
    return {"status": "ok", "bin_bytes": (bdir / "bench.bin").stat().st_size, "sections": sections}


def build_with_fallback(profile: str, dataset: str, model: str, tool: str) -> list[tuple[str, dict]]:
    """Build with the default flags; on an Xtensa literal-range failure, build again with litpools.

    Returns [(build_flags, result), ...], the default build first.
    """
    results = [("default", build(profile, dataset, model, tool))]
    if results[0][1]["status"] == "literal_range":
        results.append(("litpools", build(profile, dataset, model, tool, litpools=True)))
    return results


def flash(profile: str, port: str, litpools: bool = False) -> None:
    subprocess.run([*idf_args(profile, litpools), "-p", port, "flash"], check=True,
                   capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
