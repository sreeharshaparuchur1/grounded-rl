#!/usr/bin/env python3
"""
eval_mmml_singleturn.py
Batch single-turn evaluation of ViGoRL/Qwen-VL on tasks from a JSON file.

Reads a JSON array produced by parse_mmml_tasks.py:
    [{"image": "/path/to/img.png", "prompt": "Question?"}, ...]

For each task:
  - Runs the model (single turn)
  - Parses <think>...</think> for reasoning
  - Parses <answer>...</answer> for the final answer
  - Logs success/failure and saves a detailed JSON results file

Usage
-----
python scripts/evaluation/eval_mmml_singleturn.py \
    --model gsarch/ViGoRL-3b-Web-Grounding \
    --tasks  scripts/evaluation/mmml_baseline_tasks.json \
    --output logs/mmml_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

DIVIDER = "=" * 80


def extract_thinking(text: str) -> Optional[str]:
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch single-turn ViGoRL/Qwen-VL eval on MMML tasks"
    )
    parser.add_argument(
        "--model",
        default="gsarch/ViGoRL-3b-Web-Grounding",
        help="HF model ID or local checkpoint path",
    )
    parser.add_argument(
        "--tasks",
        default="scripts/evaluation/mmml_baseline_tasks.json",
        help="JSON task file produced by parse_mmml_tasks.py",
    )
    parser.add_argument(
        "--output",
        default="logs/mmml_results.json",
        help="Where to write the detailed per-task results JSON",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--task_start",
        type=int,
        default=0,
        help="Index of first task to run (for partial reruns)",
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=None,
        help="Exclusive end index (default: all tasks)",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    # 1. Load tasks
    # ----------------------------------------------------------------------- #
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        log.error("Task file not found: %s", tasks_path)
        sys.exit(1)

    tasks: list[dict] = json.loads(tasks_path.read_text())
    subset = tasks[args.task_start : args.task_end]
    log.info(
        "Loaded %d tasks from %s (running %d)",
        len(tasks),
        tasks_path,
        len(subset),
    )

    # ----------------------------------------------------------------------- #
    # 2. Load model + processor (once, shared across all tasks)
    # ----------------------------------------------------------------------- #
    log.info("Loading model: %s", args.model)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=12960000, min_pixels=3136
    )
    log.info("Model loaded in %.1fs", time.time() - t0)

    # ----------------------------------------------------------------------- #
    # 3. Run evaluation
    # ----------------------------------------------------------------------- #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    n_success = 0
    n_has_thinking = 0
    n_has_answer = 0

    for local_idx, task in enumerate(subset):
        global_idx = args.task_start + local_idx
        image_path = str(Path(task["image"]).expanduser())
        prompt = task["prompt"]

        log.info(DIVIDER)
        log.info("TASK %d/%d  (global #%d)", local_idx + 1, len(subset), global_idx)
        log.info("  IMAGE : %s", image_path)
        log.info("  PROMPT: %s", prompt)

        # --- Check image exists ---
        if not Path(image_path).exists():
            log.warning("  [SKIP] Image not found: %s", image_path)
            results.append(
                {
                    "task_id": global_idx,
                    "image": image_path,
                    "prompt": prompt,
                    "raw_output": None,
                    "thinking": None,
                    "answer": None,
                    "has_thinking": False,
                    "has_answer": False,
                    "success": False,
                    "error": "image_not_found",
                }
            )
            continue

        # --- Build single-turn message ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # --- Tokenise ---
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(args.device)

        # --- Generate ---
        t_gen = time.time()
        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                repetition_penalty=args.repetition_penalty,
            )
        elapsed = time.time() - t_gen

        gen_trim = gen_ids[:, inputs.input_ids.shape[1] :]
        raw_outputs = processor.batch_decode(
            gen_trim,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_output = raw_outputs[0]

        # --- Parse reasoning + answer ---
        thinking = extract_thinking(raw_output)
        answer = extract_answer(raw_output)
        has_thinking = thinking is not None
        has_answer = answer is not None
        # A response is "successful" when the model produces a final answer
        success = has_answer

        if has_thinking:
            n_has_thinking += 1
        if has_answer:
            n_has_answer += 1
        if success:
            n_success += 1

        # --- Console logging ---
        print(DIVIDER)
        print(f"TASK {global_idx} RESULT")
        print(DIVIDER)
        if has_thinking:
            print("[REASONING]")
            print(thinking)
            print()
        else:
            print("[REASONING] <none — model did not produce <think> tags>")
            print()
        if has_answer:
            print("[ANSWER]")
            print(answer)
        else:
            print("[ANSWER] <none — model did not produce <answer> tags>")
        print(f"\n[SUCCESS: {success}]  (generated in {elapsed:.1f}s)")
        print()
        if not has_answer:
            print("[RAW OUTPUT]")
            print(raw_output)
        print(DIVIDER)

        results.append(
            {
                "task_id": global_idx,
                "image": image_path,
                "prompt": prompt,
                "raw_output": raw_output,
                "thinking": thinking,
                "answer": answer,
                "has_thinking": has_thinking,
                "has_answer": has_answer,
                "success": success,
                "generation_time_s": round(elapsed, 2),
            }
        )

    # ----------------------------------------------------------------------- #
    # 4. Summary
    # ----------------------------------------------------------------------- #
    total = len(subset)
    summary = {
        "model": args.model,
        "tasks_file": str(tasks_path),
        "total_tasks": total,
        "successful": n_success,
        "success_rate": round(n_success / total, 4) if total > 0 else 0.0,
        "with_thinking": n_has_thinking,
        "thinking_rate": round(n_has_thinking / total, 4) if total > 0 else 0.0,
        "with_answer": n_has_answer,
        "answer_rate": round(n_has_answer / total, 4) if total > 0 else 0.0,
    }

    print(DIVIDER)
    print("EVALUATION SUMMARY")
    print(DIVIDER)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print(DIVIDER)

    output = {"summary": summary, "results": results}
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Results written to %s", out_path)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
