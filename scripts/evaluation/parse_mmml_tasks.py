#!/usr/bin/env python3
"""
parse_mmml_tasks.py
Convert mmml_baseline_tasks.txt (alternating image-path / prompt lines)
into a structured JSON array for batch evaluation.

Format expected in txt file:
    /path/to/image.png
    [optional blank line(s)]
    Prompt text here
    [optional blank line(s)]
    /path/to/next/image.png
    ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_tasks(txt_path: str) -> list[dict]:
    lines = Path(txt_path).read_text().splitlines()

    tasks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines
        if not line:
            i += 1
            continue

        # An image path is an absolute file path (starts with /)
        if line.startswith("/"):
            image_path = line
            i += 1

            # Advance past any blank lines between path and prompt
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i < len(lines) and not lines[i].strip().startswith("/"):
                prompt = lines[i].strip()
                tasks.append({"image": image_path, "prompt": prompt})
                i += 1
            else:
                # No prompt found before next path — skip this entry
                print(f"[WARN] No prompt found for image: {image_path}")
        else:
            i += 1

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mmml_baseline_tasks.txt to JSON"
    )
    parser.add_argument(
        "--input",
        default="scripts/evaluation/mmml_baseline_tasks.txt",
        help="Path to the input .txt file",
    )
    parser.add_argument(
        "--output",
        default="scripts/evaluation/mmml_baseline_tasks.json",
        help="Path to write the output .json file",
    )
    args = parser.parse_args()

    tasks = parse_tasks(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2))

    print(f"[INFO] Parsed {len(tasks)} tasks → {out_path}")
    for idx, t in enumerate(tasks):
        print(f"  [{idx}] image={t['image']}")
        print(f"        prompt={t['prompt'][:80]}")


if __name__ == "__main__":
    main()
