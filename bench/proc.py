"""Run external commands with a timeout that also stops their child processes."""

import os
import signal
import subprocess


def run(cmd: list[str], timeout: float, **kwargs) -> subprocess.CompletedProcess:
    """Like subprocess.run(capture_output=True, text=True), but on timeout the whole process
    group is killed, so compilers started by idf.py or a compiler driver do not keep running."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
