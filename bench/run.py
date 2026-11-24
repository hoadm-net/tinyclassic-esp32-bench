"""Flash each configuration, stream the test set over serial and record labels and cycles.

Run inside an ESP-IDF v6.1.1 environment with one board connected.

Usage: python -m bench.run --profile esp32_240 --port /dev/cu.usbserial-0001
       [--dataset ...] [--model ...] [--tool ...] [--limit N] [--force]

Output per configuration in results/raw/<profile>/:
  <dataset>__<model>__<tool>.csv   one row per test sample
  <dataset>__<model>__<tool>.json  board info, stack and heap
"""

import argparse
import json
import struct
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import serial

from bench.config import (DATA_PROCESSED, MODELS_DIR, PROFILES, RESULTS, config_id, iter_configs)
from bench.firmware import build_with_fallback, flash

SAMPLE_REPLY = struct.Struct("<ciIII")  # 'R', label, median cycles, min cycles, timed calls
END_REPLY = struct.Struct("<cIIII")     # 'M', stack size, stack free min, heap free, heap free min
REPLY_TIMEOUT_S = 120


class Board:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.2
        self.ser.dtr = self.ser.rts = False  # do not hold the chip in reset
        self.ser.open()

    def close(self):
        self.ser.close()

    def read_exact(self, n: int) -> bytes:
        data, deadline = b"", time.monotonic() + REPLY_TIMEOUT_S
        while len(data) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(f"expected {n} bytes, got {len(data)}")
            data += self.ser.read(n - len(data))
        return data

    def hello(self, timeout_s: float = 20) -> dict:
        """Ask for the READY line; boot output before it is ignored."""
        buf, deadline = b"", time.monotonic() + timeout_s
        self.ser.reset_input_buffer()
        while time.monotonic() < deadline:
            self.ser.write(b"H")
            buf += self.ser.read(512)
            start = buf.find(b"BENCH READY")
            if start >= 0 and b"\n" in buf[start:]:
                line = buf[start:buf.index(b"\n", start)].decode()
                time.sleep(0.2)
                self.ser.reset_input_buffer()  # drop replies to the extra 'H' requests
                return dict(item.split("=", 1) for item in line.split()[2:])
        raise TimeoutError("board did not answer")

    def predict(self, x: np.ndarray) -> tuple:
        self.ser.write(b"S" + x.astype("<f4").tobytes())
        tag, label, median, minimum, timed = SAMPLE_REPLY.unpack(self.read_exact(SAMPLE_REPLY.size))
        assert tag == b"R", tag
        return label, median, minimum, timed

    def end(self) -> dict:
        self.ser.write(b"E")
        tag, stack, stack_free, heap, heap_min = END_REPLY.unpack(self.read_exact(END_REPLY.size))
        assert tag == b"M", tag
        return {"stack_size": stack, "stack_free_min": stack_free,
                "heap_free": heap, "heap_free_min": heap_min}


def run_one(profile, port, baud, dataset, model, tool, limit, litpools):
    X = np.load(DATA_PROCESSED / dataset / "X_test.npy")
    y_true = np.load(DATA_PROCESSED / dataset / "y_test.npy")
    ref = (np.load(MODELS_DIR / dataset / f"{model}.ref.npy") if tool != "baseline"
           else np.zeros(len(y_true), dtype=np.int32))
    n = len(X) if limit is None else min(limit, len(X))

    flash(profile, port, litpools)
    board = Board(port, baud)
    try:
        info = board.hello()
        expected = config_id(dataset, model, tool)
        if info.get("id") != expected or int(info["nf"]) != X.shape[1]:
            raise RuntimeError(f"board runs {info}, expected {expected}")
        rows = []
        t0 = time.monotonic()
        for i in range(n):
            label, median, minimum, timed = board.predict(X[i])
            rows.append((i, label, int(ref[i]), int(y_true[i]), median, minimum, timed))
        info.update(board.end())
    finally:
        board.close()
    info.update({"profile": profile, "build_flags": "litpools" if litpools else "default",
                 "n_samples": n, "seconds": round(time.monotonic() - t0, 1),
                 "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    df = pd.DataFrame(rows, columns=["idx", "label", "ref", "y_true", "cycles_median",
                                     "cycles_min", "timed_calls"])
    return df, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--dataset", nargs="*")
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--tool", nargs="*")
    parser.add_argument("--limit", type=int, help="only the first N test samples")
    parser.add_argument("--force", action="store_true", help="rerun configurations that have results")
    args = parser.parse_args()

    out_dir = RESULTS / "raw" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, model, tool in iter_configs(args.dataset, args.model, args.tool, include_baseline=True):
        stem = out_dir / f"{dataset}__{model}__{tool}"
        cid = config_id(dataset, model, tool)
        if stem.with_suffix(".json").exists() and not args.force:
            print(f"{cid:32} done")
            continue
        builds = build_with_fallback(args.profile, dataset, model, tool)
        flags, size = builds[-1]
        if size["status"] != "ok":
            print(f"{cid:32} skip ({', '.join(r['status'] for _, r in builds)})")
            continue
        df, info = run_one(args.profile, args.port, args.baud, dataset, model, tool, args.limit,
                           litpools=flags == "litpools")
        df.to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        agree = (df.label == df.ref).mean()
        print(f"{cid:32} agree={100 * agree:.2f}% median={int(df.cycles_median.median())} cycles")


if __name__ == "__main__":
    main()
