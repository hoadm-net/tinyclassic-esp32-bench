"""Record software versions in results/env.json.

Usage: python -m bench.env   (run inside the ESP-IDF environment to include its version)
"""

import json
import platform
import subprocess
from importlib.metadata import version

from bench.config import RESULTS
from bench.firmware import idf_command

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "emlearn", "micromlgen", "m2cgen",
            "pyserial", "pyelftools"]


def command_output(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def main() -> None:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in PACKAGES},
        "host_cc": command_output(["cc", "--version"]),
        "esp_idf": command_output([*idf_command(), "--version"]),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
