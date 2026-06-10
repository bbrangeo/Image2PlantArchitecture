import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from image_process import _as_uint8_rgb, process_leaf_image
from plant_tokenizer import EOS_TOKEN, META_TOKEN, vec2token
from string_to_xml_to_vec import linked_to_recursive, xml2vec

SHARD_RANGE_PATTERN = re.compile(r"\{(\d+)\.\.(\d+)\}")


def parse_shard_range(shard_range: str) -> Tuple[int, int]:
    """Parse a shard range string like '0-160' into (start, end) inclusive."""
    start_str, end_str = shard_range.split("-", 1)
    return int(start_str), int(end_str)


def build_shard_url(url_template: str, start: int, end: int) -> str:
    """Replace the braceexpand shard range in a WebDataset URL."""
    match = SHARD_RANGE_PATTERN.search(url_template)
    if match is not None:
        width = len(match.group(1))
        replacement = f"{{{start:0{width}d}..{end:0{width}d}}}"
        return SHARD_RANGE_PATTERN.sub(replacement, url_template, count=1)
    raise ValueError(
        f"Could not find shard range pattern in dataset URL: {url_template}"
    )


def count_shards(shard_range: str) -> int:
    start, end = parse_shard_range(shard_range)
    return end - start + 1


def parse_xml_to_vec(xml_data) -> list:
    """Convert raw XML bytes/str into a plant vector."""
    if isinstance(xml_data, bytes):
        xml_str = xml_data.decode("utf-8")
    else:
        xml_str = xml_data

    root = ET.fromstring(xml_str)
    root = linked_to_recursive(root)
    plant_array = []
    xml2vec(root[0], plant_array)
    return plant_array


def process_sample(
    image,
    xml_data,
    image_processor=None,
    image_size: int = 448,
    mode: str = "train",
    process_leaf: bool = True,
    color_jitter: bool = False,
    random_crop: bool = False,
    random_erase: bool = False,
    add_sos_token: bool = False,
):
    """Process one image/XML pair into the format expected by the Trainer."""
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = np.array(image)

    image_np = _as_uint8_rgb(image_np)

    leaf_area, plant_width, plant_height, processed_img, _ = process_leaf_image(
        image_np, normalize=True, debug=False, sqaure_crop=True
    )
    plant_info = [leaf_area, plant_width, plant_height]

    if process_leaf:
        leaf_img = cv2.resize(processed_img, (image_size, image_size))
    else:
        leaf_img = cv2.resize(image_np, (image_size, image_size))

    vec = parse_xml_to_vec(xml_data)
    if len(vec) == 0:
        return None

    transform_random_resized_crop = transforms.RandomResizedCrop(
        image_size, scale=(0.8, 1.0)
    )
    transform_color_jitter = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
    )
    transform_random_erase = transforms.RandomErasing(
        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
    )

    if mode == "train":
        image = Image.fromarray(leaf_img)
        if random_crop:
            image = transform_random_resized_crop(image)
        if color_jitter:
            image = transform_color_jitter(image)
        leaf_img = np.array(image)

    image_tensor = torch.tensor(leaf_img).permute(2, 0, 1)

    if mode == "train" and random_erase:
        image_tensor = transform_random_erase(image_tensor)

    out = vec2token(vec)
    plant_info_vec = np.concatenate(([0, 0], plant_info))
    plant_info_token = vec2token([plant_info_vec])
    plant_info_token = np.concatenate(
        ([META_TOKEN], plant_info_token[1:].astype("int64"), [META_TOKEN])
    )
    if add_sos_token:
        from plant_tokenizer import SOS_TOKEN

        plant_info_token = np.concatenate(([SOS_TOKEN], plant_info_token))

    out = np.concatenate((plant_info_token, out, [EOS_TOKEN]))

    if image_processor is not None:
        pixel_values = image_processor(image_tensor, return_tensors="pt").pixel_values[0]
    else:
        pixel_values = image_tensor

    return {
        "pixel_values": pixel_values,
        "labels": out,
        "plant_info": plant_info_token,
        "plant_vec": vec,
    }


def _process_wds_dict(sample, **kwargs):
    jpeg = sample.get("jpeg")
    if jpeg is None:
        jpeg = sample.get("jpg")
    xml = sample.get("xml")
    if jpeg is None or xml is None:
        return None
    try:
        return process_sample(jpeg, xml, **kwargs)
    except Exception as exc:
        key = sample.get("__key__", "unknown")
        print(f"Skipping sample {key}: {exc}")
        return None


class CowpeaWDSIterableDataset(IterableDataset):
    """Streaming WebDataset for Cowpea image/XML pairs."""

    def __init__(
        self,
        url: str,
        image_processor=None,
        image_size: int = 448,
        mode: str = "train",
        shardshuffle: bool = True,
        shuffle_buffer: int = 1000,
        workers: int = 4,
        samples_per_epoch: Optional[int] = None,
        process_leaf: bool = True,
        color_jitter: bool = False,
        random_crop: bool = False,
        random_erase: bool = False,
        add_sos_token: bool = False,
    ):
        self.url = url
        self.image_processor = image_processor
        self.image_size = image_size
        self.mode = mode
        self.shardshuffle = shardshuffle
        self.shuffle_buffer = shuffle_buffer
        self.workers = workers
        self.samples_per_epoch = samples_per_epoch
        self.process_kwargs = {
            "image_processor": image_processor,
            "image_size": image_size,
            "mode": mode,
            "process_leaf": process_leaf,
            "color_jitter": color_jitter,
            "random_crop": random_crop,
            "random_erase": random_erase,
            "add_sos_token": add_sos_token,
        }

    def _build_pipeline(self):
        pipeline = wds.WebDataset(
            self.url,
            shardshuffle=self.shardshuffle,
            nodesplitter=wds.split_by_node,
        )
        if self.mode == "train" and self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        pipeline = pipeline.decode("rgb")
        pipeline = pipeline.map(
            lambda sample: _process_wds_dict(sample, **self.process_kwargs)
        )
        pipeline = pipeline.select(lambda sample: sample is not None)
        if self.samples_per_epoch is not None:
            pipeline = pipeline.with_epoch(self.samples_per_epoch)
        return pipeline

    def __iter__(self):
        for sample in self._build_pipeline():
            yield sample


class CowpeaWDSCollectedDataset(Dataset):
    """Collect a bounded number of WebDataset samples for evaluation/benchmarking."""

    def __init__(
        self,
        url: str,
        max_samples: int = 500,
        image_processor=None,
        image_size: int = 448,
        mode: str = "test",
        workers: int = 2,
        process_leaf: bool = True,
        add_sos_token: bool = False,
    ):
        self.samples = []
        process_kwargs = {
            "image_processor": image_processor,
            "image_size": image_size,
            "mode": mode,
            "process_leaf": process_leaf,
            "color_jitter": False,
            "random_crop": False,
            "random_erase": False,
            "add_sos_token": add_sos_token,
        }

        pipeline = (
            wds.WebDataset(url, shardshuffle=False, workersplitter=None)
            .decode("rgb")
            .map(lambda sample: _process_wds_dict(sample, **process_kwargs))
            .select(lambda sample: sample is not None)
        )

        print(f"Collecting up to {max_samples} samples from {url}...")
        for sample in pipeline:
            self.samples.append(sample)
            if len(self.samples) >= max_samples:
                break
        print(f"Collected {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
