#!/usr/bin/env python3
"""
eval_mmml_multiturn.py
Batch multi-turn evaluation of ViGoRL/Qwen-VL on tasks from a JSON file.

Follows the structure of demo/demo_multiturn.py — loads the model once and
runs the full multi-turn crop-and-zoom loop for every task.

For each task it logs:
  - Every assistant turn (raw text)
  - Each <tool_call> coordinate and the crop saved
  - The final <think> reasoning and <answer>
  - Whether the task was successful (model produced an <answer>)

Usage
-----
python scripts/evaluation/eval_mmml_multiturn.py \
    --model gsarch/ViGoRL-Multiturn-3b-Web-Grounding \
    --tasks  scripts/evaluation/mmml_baseline_tasks.json \
    --output logs/mmml_multiturn_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageDraw
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
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Helpers (mirrors demo_multiturn.py)
# --------------------------------------------------------------------------- #
def parse_coordinate(tool_text: str) -> Optional[Tuple[int, int]]:
    """Extract (x, y) from JSON inside <tool_call>…</tool_call>."""
    try:
        payload = json.loads(tool_text.strip())
        return tuple(payload["arguments"]["coordinate"])
    except Exception:
        return None


def get_point_crop(
    img: Image.Image,
    pt: Tuple[int, int],
    offset: int = 75,
    crop_size: int = 512,
    draw_dot: bool = True,
) -> Image.Image:
    """Square crop centered on pt; optionally draw a red dot."""
    x, y = pt
    w, h = img.size
    left   = max(0, x - offset)
    top    = max(0, y - offset)
    right  = min(w, x + offset)
    bottom = min(h, y + offset)
    crop = img.crop((left, top, right, bottom))
    if draw_dot:
        draw = ImageDraw.Draw(crop)
        r = 6
        draw.ellipse(
            (x - left - r, y - top - r, x - left + r, y - top + r),
            fill="red", outline="white", width=2,
        )
    crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    return crop


def extract_thinking(text: str) -> Optional[str]:
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------- #
# Single-task multi-turn loop
# --------------------------------------------------------------------------- #
def run_task(
    task_id: int,
    image_path: str,
    prompt: str,
    model,
    processor,
    device: str,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    crop_size: int,
    crop_offset: int,
    draw_dot: bool,
    crop_dir: str,
) -> dict:
    """Run one task through the multi-turn loop and return a result dict."""

    log.info(DIVIDER)
    log.info("TASK %d", task_id)
    log.info("  IMAGE : %s", image_path)
    log.info("  PROMPT: %s", prompt)

    if not Path(image_path).exists():
        log.warning("  [SKIP] Image not found: %s", image_path)
        return {
            "task_id": task_id,
            "image": image_path,
            "prompt": prompt,
            "turns": [],
            "n_turns": 0,
            "n_tool_calls": 0,
            "thinking": None,
            "answer": None,
            "has_thinking": False,
            "has_answer": False,
            "success": False,
            "forced_answer": False,
            "error": "image_not_found",
            "total_time_s": 0.0,
        }

    init_image_path = Path(image_path).expanduser()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(init_image_path)},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    turn_log: list[dict] = []
    n_tool_calls = 0
    answered = False
    final_thinking: Optional[str] = None
    final_answer: Optional[str] = None
    t_start = time.time()

    # ------------------------------------------------------------------ #
    # Multi-turn loop (mirrors demo_multiturn.py §3)
    # ------------------------------------------------------------------ #
    for turn in range(1, max_turns + 1):
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=repetition_penalty,
            )
        gen_trim = gen_ids[:, inputs.input_ids.shape[1]:]
        assistant_text = processor.batch_decode(
            gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        thinking = extract_thinking(assistant_text)
        answer   = extract_answer(assistant_text)
        tool_match = TOOL_RE.search(assistant_text)
        coord = parse_coordinate(tool_match.group(1)) if tool_match else None

        print(f"\n--- Task {task_id} | Turn {turn} ---")
        print(assistant_text)

        turn_entry: dict = {
            "turn": turn,
            "assistant_text": assistant_text,
            "thinking": thinking,
            "answer": answer,
            "tool_call_coord": list(coord) if coord else None,
            "crop_path": None,
        }

        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        })

        # ---- <answer> reached ----------------------------------------
        if answer is not None:
            final_thinking = thinking
            final_answer = answer
            answered = True
            turn_log.append(turn_entry)
            print(f"\n[Task {task_id}] Final answer reached at turn {turn}: {answer}")
            break

        # ---- <tool_call> → crop + feed back --------------------------
        if coord is not None:
            n_tool_calls += 1
            user_img = Image.open(init_image_path)
            crop = get_point_crop(
                user_img, coord,
                offset=crop_offset,
                crop_size=crop_size,
                draw_dot=draw_dot,
            )
            os.makedirs(crop_dir, exist_ok=True)
            crop_name = os.path.join(
                crop_dir, f"task{task_id}_turn{turn}_{uuid.uuid4().hex[:8]}.png"
            )
            crop.save(crop_name)
            log.info("  [Turn %d] Tool call coord=%s → crop saved: %s", turn, coord, crop_name)
            turn_entry["crop_path"] = crop_name

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<observation>\nHere is the crop of the image centered on the coordinate:\n</observation>",
                    },
                    {"type": "image", "image": crop},
                ],
            })

        turn_log.append(turn_entry)

    # ------------------------------------------------------------------ #
    # Soft-prompt if no <answer> (mirrors demo_multiturn.py §4)
    # ------------------------------------------------------------------ #
    forced_answer = False
    if not answered:
        log.info("  [Task %d] max_turns reached — sending soft prompt", task_id)
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": (
                    "<think> Based on all the information I've gathered, "
                    "I'll now provide my final answer. </think>\n<answer>"
                ),
            }],
        })

        soft_text = processor.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        soft_inputs = processor(
            text=[soft_text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            soft_ids = model.generate(
                **soft_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,   # deterministic forced answer
            )
        soft_trim = soft_ids[:, soft_inputs.input_ids.shape[1]:]
        forced_text = processor.batch_decode(
            soft_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        final_answer = forced_text.strip()
        final_thinking = None
        forced_answer = True
        answered = bool(final_answer)

        print(f"\n--- Task {task_id} | Forced final answer ---")
        print(f"<answer>{final_answer}</answer>")

    total_time = round(time.time() - t_start, 2)

    # ---- Console summary for this task -----------------------------------
    print(DIVIDER)
    print(f"TASK {task_id} SUMMARY")
    print(DIVIDER)
    if final_thinking:
        print("[REASONING]")
        print(final_thinking)
        print()
    else:
        print("[REASONING] <none in final turn>")
        print()
    print("[ANSWER]")
    print(final_answer if final_answer else "(none)")
    print(f"\n[SUCCESS: {answered}]  forced={forced_answer}  "
          f"turns={len(turn_log)}  tool_calls={n_tool_calls}  "
          f"time={total_time}s")
    print(DIVIDER)

    return {
        "task_id": task_id,
        "image": image_path,
        "prompt": prompt,
        "turns": turn_log,
        "n_turns": len(turn_log),
        "n_tool_calls": n_tool_calls,
        "thinking": final_thinking,
        "answer": final_answer,
        "has_thinking": final_thinking is not None,
        "has_answer": answered,
        "success": answered,
        "forced_answer": forced_answer,
        "error": None,
        "total_time_s": total_time,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch multi-turn ViGoRL/Qwen-VL eval on MMML tasks"
    )
    parser.add_argument(
        "--model",
        default="gsarch/ViGoRL-Multiturn-3b-Web-Grounding",
        help="HF model ID or local checkpoint path",
    )
    parser.add_argument(
        "--tasks",
        default="scripts/evaluation/mmml_baseline_tasks.json",
        help="JSON task file produced by parse_mmml_tasks.py",
    )
    parser.add_argument(
        "--output",
        default="logs/mmml_multiturn_results.json",
        help="Where to write the detailed per-task results JSON",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_turns",          type=int,   default=5)
    parser.add_argument("--max_new_tokens",     type=int,   default=512)
    parser.add_argument("--temperature",        type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--crop_size",          type=int,   default=512)
    parser.add_argument("--crop_offset",        type=int,   default=100)
    parser.add_argument("--draw_dot",           action="store_true",
                        help="Draw a red dot on each crop at the predicted coordinate")
    parser.add_argument("--crop_dir",           default="data/demo/crops",
                        help="Directory to save intermediate crop images")
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
    # Load model + processor once
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
    # Run evaluation
    # ----------------------------------------------------------------------- #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    n_success = n_has_thinking = n_has_answer = n_forced = 0
    total_turns = total_tool_calls = 0

    for local_idx, task in enumerate(subset):
        global_idx = args.task_start + local_idx
        result = run_task(
            task_id=global_idx,
            image_path=task["image"],
            prompt=task["prompt"],
            model=model,
            processor=processor,
            device=args.device,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            crop_size=args.crop_size,
            crop_offset=args.crop_offset,
            draw_dot=args.draw_dot,
            crop_dir=args.crop_dir,
        )
        results.append(result)

        if result["success"]:      n_success += 1
        if result["has_thinking"]: n_has_thinking += 1
        if result["has_answer"]:   n_has_answer += 1
        if result["forced_answer"]: n_forced += 1
        total_turns      += result["n_turns"]
        total_tool_calls += result["n_tool_calls"]

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
        "with_thinking": n_has_thinking,
        "thinking_rate": round(n_has_thinking / total, 4) if total else 0.0,
        "with_answer": n_has_answer,
        "answer_rate": round(n_has_answer / total, 4) if total else 0.0,
        "forced_answers": n_forced,
        "avg_turns": round(total_turns / total, 2) if total else 0.0,
        "avg_tool_calls": round(total_tool_calls / total, 2) if total else 0.0,
    }

    print(DIVIDER)
    print("EVALUATION SUMMARY")
    print(DIVIDER)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print(DIVIDER)

    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
