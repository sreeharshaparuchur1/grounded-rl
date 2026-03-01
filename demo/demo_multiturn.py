#!/usr/bin/env python
"""
multiturn_demo.py
Run a ViGoRL/Qwen-VL HF checkpoint in a single-image, multi-turn loop.

# 1. Visual search model
python demo/demo_multiturn.py \
    --model gsarch/ViGoRL-Multiturn-7b-Visual-Search \
    --image demo/examples/man.jpg \
    --query "What color is the man's shirt?" \
    --crop_offset 182 \
    --crop_size 672

# 2. Web grounding model
python demo/demo_multiturn.py \
    --model gsarch/ViGoRL-Multiturn-3b-Web-Grounding \
    --image demo/examples/APnews.png \
    --query "Description: check sports news" \
    --draw_dot \
    --crop_offset 100 \
    --crop_size 512
"""
from __future__ import annotations

import argparse, json, os, re, uuid
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --------------------------------------------------------------------------------
# Utility helpers
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def parse_coordinate(tool_text: str) -> Optional[Tuple[int, int]]:
    """Extract (x,y) from JSON inside <tool_call> … </tool_call>."""
    try:
        payload = json.loads(tool_text.strip())
        return tuple(payload["arguments"]["coordinate"])   # (x, y)
    except Exception:
        return None

def get_point_crop(img: Image.Image,
                   pt: Tuple[int, int],
                   offset: int = 75,
                   crop_size: int = 512,
                   draw_dot: bool = True) -> Image.Image:
    """Square crop centered on pt; optionally draw red dot."""
    x, y = pt
    w, h = img.size
    left   = max(0, x - offset);  top    = max(0, y - offset)
    right  = min(w, x + offset);  bottom = min(h, y + offset)
    crop = img.crop((left, top, right, bottom))
    if draw_dot:
        draw = ImageDraw.Draw(crop)
        r = 6
        draw.ellipse((x-left-r, y-top-r, x-left+r, y-top+r),
                     fill="red", outline="white", width=2)
    crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    return crop

# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF path or hub id, e.g. gsarch/ViGoRL-7b-Web-Grounding")
    parser.add_argument("--image", required=True, help="Path to initial image")
    parser.add_argument("--query", required=True, help="User query string")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--crop_offset", type=int, default=75)
    parser.add_argument("--draw_dot", action="store_true")
    args = parser.parse_args()

    # 1. Load model + processor --------------------------------------------------
    print("Loading checkpoint…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model,
                                              max_pixels=12960000,
                                              min_pixels=3136)

    # 2. Build initial message list ---------------------------------------------
    init_image_path = Path(args.image).expanduser()
    messages: List[dict] = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(init_image_path)},
            {"type": "text",  "text": args.query},
        ],
    }]

    # 3. Multiturn loop ----------------------------------------------------------
    print("\n=== START MULTI-TURN ===")
    assistant_text = ""
    answered = False
    for turn in range(1, args.max_turns + 1):
        # ---- existing body unchanged ----

        # Prepare inputs for this turn
        text_prompt = processor.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(text=[text_prompt],
                           images=img_inputs,
                           videos=vid_inputs,
                           padding=True,
                           return_tensors="pt").to(args.device)

        # Model generate -----------------------------------------------------
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            # repetition_penalty=1.05,
        )
        gen_trim = gen_ids[:, inputs.input_ids.shape[1]:]
        assistant_text = processor.batch_decode(
            gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"\n--- Assistant turn {turn} ---")
        print(assistant_text)

        # Append assistant message to history
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": assistant_text}]})

        # 3a. Check for <answer> => done
        if "<answer>" in assistant_text and "</answer>" in assistant_text:
            answered = True
            print("\n=== FINAL ANSWER REACHED ===")
            break

        # 3b. Check for <tool_call> => crop + feed back
        m = TOOL_RE.search(assistant_text)
        if m:
            coord = parse_coordinate(m.group(1))
            if coord:
                user_img = Image.open(init_image_path)  # always crop original
                crop = get_point_crop(user_img, coord,
                                      offset=args.crop_offset, draw_dot=args.draw_dot, crop_size=args.crop_size)
                os.makedirs("data/demo", exist_ok=True)
                crop_name = f"data/demo/crop_{turn}_{uuid.uuid4().hex[:8]}.png"
                crop.save(crop_name)
                print(f"   ↳ crop saved to {crop_name} (coord={coord})")

                # Feed crop back as observation
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "<observation>\nHere is the crop of the image centered on the coordinate:\n</observation>"},
                        {"type": "image", "image": crop},
                    ],
                })
                continue   # go to next turn
        # No tool call?  Just let next turn continue automatically.

    # --------------------------------------------------------------------------
    # 4. Soft-prompt if no <answer> was produced
    # --------------------------------------------------------------------------
    if not answered:
        print("\n>>> max_turns reached - sending soft prompt for final answer")
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<think> Based on all the information I've gathered, "
                            "I'll now provide my final answer. </think>\n<answer>"
                        ),
                    }
                ],
            }
        )

        # Build prompt WITHOUT adding a new assistant stub – we’re continuing
        soft_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        soft_inputs = processor(
            text=[soft_text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(args.device)

        soft_ids = model.generate(
            **soft_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,              # deterministic answer
        )
        soft_trim = soft_ids[:, soft_inputs.input_ids.shape[1] :]
        final_text = processor.batch_decode(
            soft_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print("\n--- Assistant (forced final) ---")
        print(f"<answer>{final_text}</answer>")
        print("\n=== SOFT-PROMPT COMPLETE ===")

if __name__ == "__main__":
    main()