#!/usr/bin/env python3
"""
eval_mmml_singleturn_base.py
Batch single-turn evaluation using the base Qwen2.5-VL-3B-Instruct model
(no ViGoRL weights) on tasks from a JSON file.

Unlike the ViGoRL eval, the base model does not produce <think>/<answer> tags,
so success is defined as a non-empty response. <think>/<answer> are still parsed
and logged in case they appear.

Usage
-----
python scripts/evaluation/eval_mmml_singleturn_base.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks  scripts/evaluation/mmml_baseline_tasks.json \
    --output logs/mmml_base_results.json
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
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DIVIDER = "=" * 80
THINK_RE  = re.compile(r"<think>(.*?)</think>",   re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


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
        description="Batch single-turn base Qwen2.5-VL-3B eval on MMML tasks"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model ID or local checkpoint path",
    )
    parser.add_argument(
        "--tasks",
        default="scripts/evaluation/mmml_baseline_tasks.json",
        help="JSON task file produced by parse_mmml_tasks.py",
    )
    parser.add_argument(
        "--output",
        default="logs/mmml_base_results.json",
        help="Where to write the detailed per-task results JSON",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_new_tokens",     type=int,   default=1024)
    parser.add_argument("--temperature",        type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end",   type=int, default=None)
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    # Load tasks
    # ----------------------------------------------------------------------- #
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        log.error("Task file not found: %s", tasks_path)
        sys.exit(1)

    tasks: list[dict] = json.loads(tasks_path.read_text())
    subset = tasks[args.task_start : args.task_end]
    log.info("Loaded %d tasks (running %d)", len(tasks), len(subset))

    # ----------------------------------------------------------------------- #
    # Load model + processor
    # sdpa attention: works without flash-attention being compiled
    # ----------------------------------------------------------------------- #
    log.info("Loading model: %s", args.model)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",   # no flash-attn required
    )
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=12960000, min_pixels=3136
    )
    log.info("Model loaded in %.1fs", time.time() - t0)

    # ----------------------------------------------------------------------- #
    # Run evaluation
    # ----------------------------------------------------------------------- #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    n_success = n_has_thinking = n_has_answer = 0

    for local_idx, task in enumerate(subset):
        global_idx = args.task_start + local_idx
        image_path = str(Path(task["image"]).expanduser())
        prompt = task["prompt"]

        log.info(DIVIDER)
        log.info("TASK %d/%d  (global #%d)", local_idx + 1, len(subset), global_idx)
        log.info("  IMAGE : %s", image_path)
        log.info("  PROMPT: %s", prompt)

        # --- Image existence check ---
        if not Path(image_path).exists():
            log.warning("  [SKIP] Image not found: %s", image_path)
            results.append({
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
                "generation_time_s": 0.0,
            })
            continue

        # --- Build single-turn message (same as demo_singleturn.py) ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

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

        gen_trim = gen_ids[:, inputs.input_ids.shape[1]:]
        raw_output = processor.batch_decode(
            gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # --- Parse structured tags if present ---
        thinking = extract_thinking(raw_output)
        answer   = extract_answer(raw_output)
        has_thinking = thinking is not None
        has_answer   = answer is not None

        # Base model success = non-empty output
        success = bool(raw_output.strip())

        if has_thinking: n_has_thinking += 1
        if has_answer:   n_has_answer += 1
        if success:      n_success += 1

        # --- Console output — always print raw response for base model ---
        print(DIVIDER)
        print(f"TASK {global_idx} RESULT  [{elapsed:.1f}s]")
        print(DIVIDER)
        if has_thinking:
            print("[REASONING]")
            print(thinking)
            print()
        if has_answer:
            print("[ANSWER TAG]")
            print(answer)
            print()
        print("[RAW OUTPUT]")
        print(raw_output)
        print(f"\n[SUCCESS: {success}]")
        print(DIVIDER)

        results.append({
            "task_id": global_idx,
            "image": image_path,
            "prompt": prompt,
            "raw_output": raw_output,
            "thinking": thinking,
            "answer": answer,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "success": success,
            "error": None,
            "generation_time_s": round(elapsed, 2),
        })

    # ----------------------------------------------------------------------- #
    # Summary
    # ----------------------------------------------------------------------- #
    total = len(subset)
    summary = {
        "model": args.model,
        "tasks_file": str(tasks_path),
        "total_tasks": total,
        "successful": n_success,
        "success_rate": round(n_success / total, 4) if total else 0.0,
        "with_thinking_tags": n_has_thinking,
        "with_answer_tags": n_has_answer,
    }

    print(DIVIDER)
    print("EVALUATION SUMMARY")
    print(DIVIDER)
    for k, v in summary.items():
        print(f"  {k:<24}: {v}")
    print()
    print("  Per-task results:")
    for r in results:
        status  = "OK " if r["success"] else "---"
        raw_preview = (r["raw_output"] or "(none)")[:70].replace("\n", " ")
        t_sec = r.get("generation_time_s", "?")
        print(f"  [{status}] task {r['task_id']:>2} | {raw_preview}  [{t_sec}s]")
    print(DIVIDER)

    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
