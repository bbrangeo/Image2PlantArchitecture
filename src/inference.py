#!/usr/bin/env python3
"""Run inference with a trained PlantArchitectureModel."""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, "..")
sys.path.insert(0, project_dir)
sys.path.insert(0, script_dir)

from image_process import process_leaf_image
from linked_xml_to_recursive_xml import recursive_to_linked, pretty_print_xml
from models.model import PlantArchitectureModel
from plant_dataset import load_sideview_images
from plant_tokenizer import EOS_TOKEN, META_TOKEN, PAD_TOKEN, SOS_TOKEN, token2vec, vec2token
from string_to_xml_to_vec import vec2xml

DEFAULT_CHECKPOINT = "heesup/dinov2-small_448_Sideview_gpt2-medium"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plant architecture XML from an image."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Hugging Face checkpoint or local path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--image",
        help="Path to a single image (tiled into a 2x2 grid for Sideview checkpoints)",
    )
    parser.add_argument(
        "--images-dir",
        help="Directory containing side-view images (used with --image-stem)",
    )
    parser.add_argument(
        "--image-stem",
        help="Base image name for side-view inference, e.g. plot_0_00 for plot_0_00.jpeg",
    )
    parser.add_argument(
        "--output",
        default="generated_plant.xml",
        help="Output XML file path (default: generated_plant.xml)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size (default: inferred from checkpoint, 224 or 448)",
    )
    parser.add_argument(
        "--side-view",
        action="store_true",
        help="Force side-view 2x2 layout (auto-enabled for Sideview checkpoints)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--leaf-area",
        type=float,
        default=None,
        help="Optional leaf area metadata override",
    )
    parser.add_argument(
        "--plant-width",
        type=float,
        default=None,
        help="Optional plant width metadata override",
    )
    parser.add_argument(
        "--plant-height",
        type=float,
        default=None,
        help="Optional plant height metadata override",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096 * 2,
        help="Maximum generation length (default: 8192)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for generation (default: 1.1)",
    )
    return parser.parse_args()


def infer_image_size(checkpoint_path, image_size):
    if image_size is not None:
        return image_size
    if "448" in checkpoint_path:
        return 448
    if "224" in checkpoint_path:
        return 224
    return 448


def single_image_to_sideview_grid(image_path, image_size):
    """Build a 2x2 side-view grid from one image (for Sideview checkpoints)."""
    image = np.array(Image.open(image_path).convert("RGB"))
    leaf_area, plant_width, plant_height, processed_img, _ = process_leaf_image(
        image, normalize=True, debug=False, sqaure_crop=True
    )
    quadrant = cv2.resize(processed_img, (image_size // 2, image_size // 2))
    total_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    total_img[: image_size // 2, : image_size // 2] = quadrant
    total_img[: image_size // 2, image_size // 2 :] = quadrant
    total_img[image_size // 2 :, : image_size // 2] = quadrant
    total_img[image_size // 2 :, image_size // 2 :] = quadrant
    return total_img, [leaf_area, plant_width, plant_height]


def wants_sideview_layout(args):
    return args.side_view or "Sideview" in args.checkpoint


def load_image_and_metadata(args, image_size):
    if args.images_dir and args.image_stem:
        image_file_name = f"{args.image_stem}.jpeg"
        image, plant_info = load_sideview_images(
            args.images_dir,
            image_file_name,
            image_size,
            process_leaf=True,
            flip_test=False,
        )
    elif args.image:
        if wants_sideview_layout(args):
            image, plant_info = single_image_to_sideview_grid(args.image, image_size)
        else:
            image = np.array(Image.open(args.image).convert("RGB"))
            leaf_area, plant_width, plant_height, processed_img, _ = process_leaf_image(
                image, normalize=True, debug=False, sqaure_crop=True
            )
            plant_info = [leaf_area, plant_width, plant_height]
            image = cv2.resize(processed_img, (image_size, image_size))
    else:
        raise ValueError(
            "Provide --image for a single image, or --images-dir and --image-stem for multi-view side-view inference"
        )

    if args.leaf_area is not None:
        plant_info[0] = args.leaf_area
    if args.plant_width is not None:
        plant_info[1] = args.plant_width
    if args.plant_height is not None:
        plant_info[2] = args.plant_height

    return image, np.asarray(plant_info, dtype=np.float64)


def build_plant_info_tokens(plant_info):
    plant_info_vec = np.concatenate(([0, 0], plant_info))
    plant_info_token = vec2token([plant_info_vec])
    plant_info_token = np.concatenate(
        ([META_TOKEN], plant_info_token[1:].astype("int64"), [META_TOKEN])
    )
    return plant_info_token


def strip_generation_prefix(tokens, plant_info_tokens):
    """Remove SOS and/or plant metadata prefix from generated tokens."""
    tokens = np.asarray(tokens, dtype=np.int64)
    plant_info_tokens = np.asarray(plant_info_tokens, dtype=np.int64)
    info_len = len(plant_info_tokens)
    if info_len == 0:
        return tokens

    if len(tokens) >= info_len and np.array_equal(tokens[:info_len], plant_info_tokens):
        return tokens[info_len:]

    if (
        len(tokens) > info_len
        and tokens[0] == SOS_TOKEN
        and np.array_equal(tokens[1 : info_len + 1], plant_info_tokens)
    ):
        return tokens[info_len + 1 :]

    start = 1 if len(tokens) > 0 and tokens[0] == SOS_TOKEN else 0
    return tokens[start + info_len :]


def preprocess_image(image, image_processor):
    image_tensor = torch.tensor(image).permute(2, 0, 1)
    return image_processor(image_tensor, return_tensors="pt").pixel_values


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_size = infer_image_size(args.checkpoint, args.image_size)

    print(f"Loading model from {args.checkpoint}")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = PlantArchitectureModel.from_pretrained(args.checkpoint, torch_dtype=dtype).to(device)
    model.eval()

    encoder_name = model.encoder.config._name_or_path
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    image_processor.crop_size["width"] = image_size
    image_processor.crop_size["height"] = image_size
    image_processor.size["shortest_edge"] = image_size

    image, plant_info = load_image_and_metadata(args, image_size)
    plant_info_tokens = build_plant_info_tokens(plant_info)

    pixel_values = preprocess_image(image, image_processor).to(device, dtype=dtype)
    plant_info_tensor = (
        torch.tensor(plant_info_tokens, dtype=torch.long).unsqueeze(0).to(device)
    )

    print("Generating plant architecture...")
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.amp.autocast("cuda"):
                result = model.generate(
                    pixel_values,
                    decoder_start_token_id=SOS_TOKEN,
                    decoder_input_ids=plant_info_tensor,
                    eos_token_id=EOS_TOKEN,
                    pad_token_id=PAD_TOKEN,
                    max_length=args.max_length,
                    repetition_penalty=args.repetition_penalty,
                    use_cache=True,
                )
        else:
            result = model.generate(
                pixel_values,
                decoder_start_token_id=SOS_TOKEN,
                decoder_input_ids=plant_info_tensor,
                eos_token_id=EOS_TOKEN,
                pad_token_id=PAD_TOKEN,
                max_length=args.max_length,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

    result_tokens = strip_generation_prefix(
        result.squeeze().cpu().numpy(), plant_info_tokens
    )
    plant_vec = [line for line in token2vec(result_tokens) if line is not None]
    if not plant_vec:
        raise ValueError(
            "Model output could not be converted to a plant architecture. "
            "The generated token sequence is empty or missing structure tokens."
        )

    plant_xml = vec2xml(plant_vec)
    plant_xml = recursive_to_linked(plant_xml)
    plant_xml_str = pretty_print_xml(plant_xml)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(plant_xml_str)

    print(f"Saved generated architecture to {output_path}")


if __name__ == "__main__":
    main()
