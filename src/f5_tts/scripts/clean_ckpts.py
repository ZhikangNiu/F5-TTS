"""Clean up training checkpoints to save disk space.

Rules:
- model_last.pt: skip (never touch)
- model_<N>.pt where N % 100000 == 0: keep only ema_model_state_dict and update
- model_<N>.pt where N % 100000 != 0: delete

Usage:
    python src/f5_tts/scripts/clean_ckpts.py /path/to/ckpts/experiment/         # dry-run
    python src/f5_tts/scripts/clean_ckpts.py /path/to/ckpts/experiment/ --execute  # actually do it
"""

import argparse
import logging
import os

import torch


def parse_step(filename):
    """Extract step number from checkpoint filename like model_150000.pt"""
    return int(filename.split("_")[1].split(".")[0])


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(description="Clean up training checkpoints to save disk space.")
    parser.add_argument("ckpt_dir", type=str, help="Path to checkpoint directory")
    parser.add_argument(
        "--execute", action="store_true", help="Actually perform deletions and trimming (default: dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ckpt_dir = args.ckpt_dir
    if not os.path.isdir(ckpt_dir):
        logging.error("%s is not a directory", ckpt_dir)
        return

    # Scan for model_*.pt files, excluding model_last.pt
    all_files = sorted(os.listdir(ckpt_dir))
    ckpt_files = []
    for filename in all_files:
        if filename.startswith("model_") and filename.endswith(".pt") and filename != "model_last.pt":
            try:
                parse_step(filename)
                ckpt_files.append(filename)
            except (ValueError, IndexError):
                logging.warning("Skipping unrecognized file: %s", filename)

    if not ckpt_files:
        logging.info("No checkpoint files found to process.")
        return

    to_delete = []
    to_trim = []
    for filename in ckpt_files:
        step = parse_step(filename)
        if step % 100000 == 0:
            to_trim.append(filename)
        else:
            to_delete.append(filename)

    # Print plan
    delete_size = sum(os.path.getsize(os.path.join(ckpt_dir, filename)) for filename in to_delete)
    trim_sizes = {filename: os.path.getsize(os.path.join(ckpt_dir, filename)) for filename in to_trim}

    logging.info("Checkpoint directory: %s", ckpt_dir)
    logging.info("Found %d checkpoint(s)\n", len(ckpt_files))

    if to_delete:
        logging.info("DELETE (%d files, %s):", len(to_delete), human_size(delete_size))
        for filename in to_delete:
            size = os.path.getsize(os.path.join(ckpt_dir, filename))
            logging.info("  %s  (%s)", filename, human_size(size))
    else:
        logging.info("DELETE: (none)")

    if to_trim:
        logging.info("TRIM to ema_model_state_dict + update (%d files):", len(to_trim))
        for filename in to_trim:
            logging.info("  %s  (%s)", filename, human_size(trim_sizes[filename]))
    else:
        logging.info("TRIM: (none)")

    if not args.execute:
        logging.warning("Dry-run mode. Add --execute to perform the above operations.")
        return

    # Execute
    logging.info("Executing...\n")
    total_freed = 0

    for filename in to_delete:
        path = os.path.join(ckpt_dir, filename)
        size = os.path.getsize(path)
        os.remove(path)
        total_freed += size
        logging.info("  Deleted %s  (%s)", filename, human_size(size))

    for filename in to_trim:
        path = os.path.join(ckpt_dir, filename)
        old_size = os.path.getsize(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        trimmed_ckpt = {
            "ema_model_state_dict": ckpt["ema_model_state_dict"],
            "update": ckpt["update"],
        }
        torch.save(trimmed_ckpt, path)
        new_size = os.path.getsize(path)
        total_freed += old_size - new_size
        logging.info("  Trimmed %s  (%s -> %s)", filename, human_size(old_size), human_size(new_size))

    logging.info("Total space freed: %s", human_size(total_freed))


if __name__ == "__main__":
    main()
