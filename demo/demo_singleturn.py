#!/usr/bin/env python
"""
singleturn_demo.py
Run a ViGoRL/Qwen-VL HF checkpoint on a *single* image / single-turn query.

Examples
--------

# 1. Spatial reasoning
python demo/demo_singleturn.py \
    --model gsarch/ViGoRL-7b-Spatial \
    --image demo/examples/LivingRoom.jpg \
    --query "What is above the blue lamp?"

# 2. Web grounding
python demo/demo_singleturn.py \
    --model gsarch/ViGoRL-7b-Web-Grounding \
    --image demo/examples/APnews.png \
    --query "Description: check sports news"

# 3. Web action
python demo/demo_singleturn.py \
    --model gsarch/ViGoRL-7b-Web-Action \
    --image demo/examples/osclass_page.png \
    --query "OBJECTIVE: Identify the insect in the picture. Leave a comment with the title \"Questions\" and text containing the insect's identity, with the purpose of confirming with the seller.\n\nPREVIOUS ACTIONS: \n1. type  [INPUT] [] [Questions]\n\nIMAGE:\n"

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-turn ViGoRL/Qwen-VL demo (image + text → answer)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model ID or local path, e.g. gsarch/ViGoRL-7b-Spatial",
    )
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--query", required=True, help="User query string")
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device for inputs"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    # 1. Load model + processor
    # ----------------------------------------------------------------------- #
    print(f"Loading checkpoint [{args.model}] …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=12960000, min_pixels=3136
    )

    # ----------------------------------------------------------------------- #
    # 2. Build messages list (single turn = one user message)
    # ----------------------------------------------------------------------- #
    img_path = str(Path(args.image).expanduser())
    messages: List[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": args.query},
            ],
        }
    ]

    # ----------------------------------------------------------------------- #
    # 3. Prepare model inputs
    # ----------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------- #
    # 4. Generate
    # ----------------------------------------------------------------------- #
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        repetition_penalty=args.repetition_penalty,
    )
    gen_trim = gen_ids[:, inputs.input_ids.shape[1] :]
    output_text = processor.batch_decode(
        gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # ----------------------------------------------------------------------- #
    # 5. Pretty-print
    # ----------------------------------------------------------------------- #
    print("=" * 80)
    print("MODEL OUTPUT:")
    print("=" * 80)
    for i, txt in enumerate(output_text):
        if len(output_text) > 1:
            print(f"\n--- Response {i+1} ---")
        print(txt)
    print("=" * 80)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
