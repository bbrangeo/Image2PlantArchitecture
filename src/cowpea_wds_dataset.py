import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

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
HF_DATASET_URL_PREFIX = re.compile(
    r"https://huggingface\.co/datasets/(?P<repo_id>.+?)/resolve/(?P<revision>[^/]+)/"
)
# Public WDS repo currently ships 40 shards (shard-000000 .. shard-000039).
HF_WDS_DEFAULT_MAX_SHARD = 39
HF_WDS_DEFAULT_TOTAL_SAMPLES = 79_560  # ~1990 samples/shard × 40 shards


def is_remote_wds_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _shard_filenames_in_url(url: str) -> List[str]:
    match = SHARD_RANGE_PATTERN.search(url)
    if match is None:
        return [url.rsplit("/", 1)[-1]]
    start, end = int(match.group(1)), int(match.group(2))
    width = len(match.group(1))
    basename = url.rsplit("/", 1)[-1]
    brace = match.group(0)
    before, after = basename.split(brace, 1)
    return [f"{before}{i:0{width}d}{after}" for i in range(start, end + 1)]


def _remote_shard_url(url: str, shard_filename: str) -> str:
    return f"{url.rsplit('/', 1)[0]}/{shard_filename}"


def _local_braceexpand_url(url: str, cache_dir: str) -> str:
    basename = url.rsplit("/", 1)[-1]
    match = SHARD_RANGE_PATTERN.search(basename)
    if match is None:
        return os.path.join(cache_dir, basename)
    start, end = match.group(1), match.group(2)
    before = basename[: match.start()]
    after = basename[match.end() :]
    return os.path.join(cache_dir, f"{before}{{{start}..{end}}}{after}")


def _download_http_file(url: str, dest_path: str, retries: int = 5) -> None:
    tmp_path = f"{dest_path}.partial"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with open(tmp_path, "wb") as out_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
            os.replace(tmp_path, dest_path)
            return
        except Exception as exc:
            last_error = exc
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"Download attempt {attempt}/{retries} failed for {url}: {exc}")
    raise OSError(f"Could not download {url}") from last_error


def parse_hf_dataset_url(url: str) -> Optional[Tuple[str, str]]:
    match = HF_DATASET_URL_PREFIX.match(url)
    if match is None:
        return None
    return match.group("repo_id"), match.group("revision")


def list_hf_dataset_shards(repo_id: str, revision: str = "main") -> List[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.startswith("shard-") and f.endswith(".tar"))


def _download_hf_dataset_shard(
    repo_id: str, revision: str, cache_dir: str, filename: str
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=cache_dir,
    )


def cache_wds_url(url: str, cache_dir: str) -> str:
    """Download remote shard(s) to cache_dir and return a local braceexpand URL."""
    if not is_remote_wds_url(url):
        return url

    os.makedirs(cache_dir, exist_ok=True)
    shard_names = _shard_filenames_in_url(url)
    hf_ref = parse_hf_dataset_url(url)

    if hf_ref is not None:
        repo_id, revision = hf_ref
        available = set(list_hf_dataset_shards(repo_id, revision))
        missing = [name for name in shard_names if name not in available]
        if missing:
            avail_nums = sorted(
                int(name.replace("shard-", "").replace(".tar", "")) for name in available
            )
            raise FileNotFoundError(
                f"{len(missing)} shard(s) not in Hugging Face dataset {repo_id} "
                f"(e.g. {missing[0]}). Available shard indices: "
                f"{avail_nums[0]}..{avail_nums[-1]} ({len(available)} shards). "
                "Update --dataset_url brace range and --train_shards/--val_shards/--test_shards."
            )
        print(
            f"Caching {len(shard_names)} WebDataset shard(s) from {repo_id} "
            f"under {cache_dir}..."
        )
        for shard_name in shard_names:
            dest_path = os.path.join(cache_dir, shard_name)
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
                continue
            print(f"  downloading {shard_name}")
            _download_hf_dataset_shard(repo_id, revision, cache_dir, shard_name)
    else:
        print(f"Caching {len(shard_names)} WebDataset shard(s) under {cache_dir}...")
        for shard_name in shard_names:
            dest_path = os.path.join(cache_dir, shard_name)
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
                continue
            remote_url = _remote_shard_url(url, shard_name)
            print(f"  downloading {shard_name}")
            _download_http_file(remote_url, dest_path)

    local_url = _local_braceexpand_url(url, cache_dir)
    print(f"Using local WebDataset URL: {local_url}")
    return local_url


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
