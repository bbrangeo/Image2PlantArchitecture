# Image2PlantArchitecture

A Vision Language Model (VLM) for generating XML-based, organ-level 3D plant architecture representations from images. This project focuses on cowpea (Vigna unguiculata) and utilizes synthetic data generated via the Helios 3D plant simulator to train a model capable of reconstructing detailed structural parameters from 2D imagery.

## Overview

Image2PlantArchitecture treats the task of 3D plant reconstruction as a sequence generation problem. By converting procedural XML descriptions of plant morphology into a specialized token sequence, a vision encoder-decoder model (DINOv2 + GPT-2) can be trained to "translate" images into structural code.

Key contributions:
- A specialized **plant architecture tokenizer** that preserves hierarchical relationships between organs (shoots, internodes, petioles, leaves).
- An end-to-end pipeline for predicting organ-level geometric parameters (length, radius, angles) from single or multi-view images.
- Demonstration that VLMs can implicitly learn bulk plant traits (leaf area, leaf count) more accurately than traditional feature regression by understanding internal 3D structure.

## Paper Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{yun2025vision,
  title={A Vision Language Model for Generating XML-based Organ-level Plant Architecture Representations of Cowpea from Simulated Images},
  author={Heesup Yun and Isaac Kazuo Uyehara and Ioannis Droutsas and Earl Ranario and Christine H. Diepenbrock and Brian N. Bailey and J. Mason Earles},
  journal={arXiv preprint arXiv:2603.22622},
  year={2026},
  url={https://arxiv.org/abs/2603.22622}
}
```

## Repository Structure

```text
├── CowpeaSimulator/      # C++ plant simulation code and Helios integration
├── models/               # PyTorch model definitions (Dinov2 + GPT-2)
├── src/                  # Python source code for training and evaluation
│   ├── train.py          # Main training script (local dataset)
│   ├── train_2.py        # Training from Hugging Face WebDataset
│   ├── inference.py      # CLI inference (image → XML)
│   ├── cowpea_wds_dataset.py  # WebDataset adapter for Cowpea-Architecture-XML
│   ├── plant_dataset.py  # Data loading and augmentation
│   ├── plant_tokenizer.py# XML to token conversion logic
│   └── test.ipynb        # Inference and visualization notebook
├── environment_cuda.yml  # Conda environment for GPU training
└── run_experiments.sh    # Shell script for batch experiments
```

## Installation

### 1. Python Environment (GPU / Linux)

Tested on the **tatanka** GPU node (CUDA 12.8, Python 3.10).

```bash
micromamba env create -f environment_cuda.yml -n .env
micromamba activate .env

# CUDA 12.8 hosts: reinstall the matched torch stack (avoids torchvision mismatch)
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# sanity check
python -c "import torch, torchvision; from open_clip.model import TextTransformer; \
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

`environment_cuda.yml` installs torch via **pip only** (not conda) to avoid solver conflicts and mixed pip/conda builds. For other CUDA tags (`cu126`, `cu121`, `cpu`), see [pytorch.org previous versions](https://pytorch.org/get-started/previous-versions/).

### 2. Build the Helios Simulator
The simulator is required for re-rendering generated XML files into 3D models:
```bash
cd CowpeaSimulator
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSKIP_INSTALL_ALL=ON .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j$(nproc)
```

## Usage

### Training
To train the model using default parameters:
```bash
python src/train.py \
    --dataset_path /path/to/dataset \
    --encoder_checkpoint facebook/dinov2-small \
    --decoder_checkpoint gpt2-medium \
    --image_size 448 \
    --batch_size 4 \
    --epoch 4
```

### Training from Hugging Face WebDataset

Train from the [Cowpea-Architecture-XML-WDS](https://huggingface.co/datasets/heesup/Cowpea-Architecture-XML-WDS) shards with `src/train_2.py`.

**Smoke test** (1 shard, 20 test samples, 1 epoch — validated on tatanka):

```bash
python src/train_2.py \
    --dataset_url "https://huggingface.co/datasets/heesup/Cowpea-Architecture-XML-WDS/resolve/main/shard-{000000..000200}.tar" \
    --train_shards 0-0 --val_shards 0-0 --test_shards 0-0 \
    --encoder_checkpoint facebook/dinov2-small \
    --decoder_checkpoint gpt2-medium \
    --image_size 448 \
    --batch_size 2 \
    --epoch 1 \
    --debug True
```

**Full training** (default shard splits: `0-160` train, `161-180` val, `181-200` test):

```bash
python src/train_2.py \
    --dataset_url "https://huggingface.co/datasets/heesup/Cowpea-Architecture-XML-WDS/resolve/main/shard-{000000..000200}.tar" \
    --encoder_checkpoint facebook/dinov2-small \
    --decoder_checkpoint gpt2-medium \
    --image_size 448 \
    --batch_size 4 \
    --epoch 4
```

Checkpoints are saved under `log/<date>/<exp_name>/checkpoints/`. Training resumes automatically if that folder already exists. Override shard splits with `--train_shards`, `--val_shards`, and `--test_shards`.

Remote shards are **cached locally by default** under `data/cowpea_wds/` (via `huggingface_hub`) to avoid `curl` exit 56 during training. Disable with `--stream_wds True` or set `--wds_cache_dir ""`.

For a full dataset copy up front:

```bash
huggingface-cli download heesup/Cowpea-Architecture-XML-WDS \
  --repo-type dataset --local-dir ./data/cowpea_wds
# then: --dataset_url "./data/cowpea_wds/shard-{000000..000200}.tar"
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `operator torchvision::nms does not exist` | Reinstall matched torch/torchvision/torchaudio via pip (see Installation §1). Never mix pip `torch` with conda `torchvision`. |
| `No module named 'open_clip'` | `pip install open-clip-torch==2.24.0` |
| `torch.torch_version` missing / broken import | Env corrupted — recreate with `environment_cuda.yml` and reinstall the cu128 stack. |
| `torch.load` / CVE-2025-32434 on resume | Requires `torch>=2.6`; or delete `optimizer.pt` / `scheduler.pt` in the checkpoint folder. |
| `EncoderDecoderCache` error at eval | Update to latest `main` (`PlantArchitectureTrainer` fix). |
| WebDataset `curl` exit 56 | Default `--wds_cache_dir data/cowpea_wds` caches shards before training. Or pre-download manually (see above). |
| Inference: `Depth & Organ is not defined` / empty XML | Checkpoint too early (smoke test) or wrong image layout. Use `heesup/dinov2-small_448_Sideview_gpt2-medium` or a trained `.../results` checkpoint. |

### Inference

Use `src/inference.py` to generate plant architecture XML from an image.

**Single image** (default Hugging Face checkpoint):

```bash
python src/inference.py \
    --image path/to/plant.jpeg \
    --output generated_plant.xml
```

**Sideview model** (2×2 grid from one image — auto-detected when checkpoint name contains `Sideview`):

```bash
python src/inference.py \
    --checkpoint heesup/dinov2-small_448_Sideview_gpt2-medium \
    --image path/to/plant.jpeg \
    --output generated_plant.xml
```

**Local checkpoint** (use `.../results` after full training, not early `checkpoint-N`):

```bash
python src/inference.py \
    --checkpoint log/20250430_TrainValTestByShard/dinov2-small_448_WDS_gpt2-medium/results \
    --image cowpea.jpeg \
    --output generated_plant.xml
```

**Multi-view side-view** (four images in a directory, stem without extension):

```bash
python src/inference.py \
    --checkpoint heesup/dinov2-small_448_Sideview_gpt2-medium \
    --images-dir /path/to/sideview_images \
    --image-stem plot_0_00 \
    --output generated_plant.xml
```

Optional flags: `--image-size 448`, `--device cuda`, `--num-beams 5`, `--leaf-area`, `--plant-width`, `--plant-height`, `--max-length`, `--repetition-penalty`.

#### Python API

You can also call `PlantArchitectureModel` directly:

```python
import torch
import cv2
import os
from models.model import PlantArchitectureModel
from transformers import AutoImageProcessor
from src.plant_tokenizer import token2vec, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from src.string_to_xml_to_vec import vec2xml
from src.linked_xml_to_recursive_xml import recursive_to_linked, pretty_print_xml

# Load model and processor
checkpoint_path = "heesup/dinov2-small_448_Sideview_gpt2-medium"
model = PlantArchitectureModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16).to("cuda")
model.eval()

# Prepare input image and metadata
# pixel_values = ... (preprocessed image tensor)
# plant_info = ... (metadata tensor)

############## Generate
with torch.no_grad():
    with torch.cuda.amp.autocast():
        result = model.generate(
            pixel_values,
            decoder_start_token_id=SOS_TOKEN,
            decoder_input_ids=plant_info,
            eos_token_id=EOS_TOKEN,
            pad_token_id=PAD_TOKEN,
            max_length=4096 * 2,
            repetition_penalty=1.1,
            use_cache=True
        )
        # Skip metadata tokens
        result_tokens = result.squeeze().cpu().numpy()[6:]

# Convert tokens back to XML
plant_vec = token2vec(result_tokens)
plant_xml = vec2xml(plant_vec)
plant_xml = recursive_to_linked(plant_xml)
plant_xml_str = pretty_print_xml(plant_xml)

# Save the generated architecture
with open("generated_plant.xml", "w") as f:
    f.write(plant_xml_str)
```

## Evaluation Results

Our best-performing model (ViT-B + GPT-2-Medium) achieved the following scores on synthetic test data:
- **BLEU-4**: 94.00%
- **ROUGE-L**: 0.5182
- **Leaf Area MAPE**: 3.2%
- **Leaf Count MAPE**: 4.1%

The model shows high robustness in predicting internal structures that are often occluded in 2D views.

## License

This project is supported by the Bill & Melinda Gates Foundation. Code is released under the MIT License.

## Contact

For questions or collaborations, please contact Heesup Yun at hspyun@ucdavis.edu.
