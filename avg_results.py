import sys
from pathlib import Path


METRICS = [
    ("WER", "_wer_results.jsonl"),
    ("SIM", "_sim_results.jsonl"),
    ("UTMOS", "_utmos_results.jsonl"),
]


def read_metric(filepath):
    lines = filepath.read_text().strip().splitlines()
    return float(lines[-1].split(": ")[1])


def main():
    exp_path = Path(sys.argv[1])
    test_sets = sorted(p for p in exp_path.iterdir() if p.is_dir())

    for ts in test_sets:
        seeds = sorted(p for p in ts.iterdir() if p.is_dir() and p.name.startswith("seed"))
        if not seeds:
            continue
        print(f"\n=== {ts.name} ===")
        for label, fname in METRICS:
            values = {}
            for s in seeds:
                f = s / fname
                if f.exists():
                    values[s.name] = read_metric(f)
            if not values:
                continue
            avg = sum(values.values()) / len(values)
            details = ", ".join(f"{k}: {v:.4f}" for k, v in values.items())
            print(f"  {label:5s}: {avg:.4f} | {details}")


if __name__ == "__main__":
    main()
