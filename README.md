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
│   ├── cowpea_wds_dataset.py  # WebDataset adapter for Cowpea-Architecture-XML
│   ├── plant_dataset.py  # Data loading and augmentation
│   ├── plant_tokenizer.py# XML to token conversion logic
│   └── test.ipynb        # Inference and visualization notebook
├── environment_cuda.yml  # Conda environment for GPU training
└── run_experiments.sh    # Shell script for batch experiments
```

## Installation

### 1. Python Environment
Create and activate the conda environment:
```bash
conda env create -f environment_cuda.yml -p .env
conda activate .env
```

### 2. Build the Helios Simulator
The simulator is required for re-rendering generated XML files into 3D models:
```bash
cd CowpeaSimulator
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSKIP_INSTALL_ALL=ON .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j$(nproc)
```

### Troubleshooting

**`operator torchvision::nms does not exist` / `Could not import AutoImageProcessor`**

`torch` and `torchvision` must be installed as a matched pair. This often happens when `torch` was upgraded via **pip** but `torchvision` is still an old **conda/micromamba** build.

Check versions:

```bash
micromamba activate .env
python -c "import torch; print('torch', torch.__version__)"
python -c "import torchvision; print('torchvision', torchvision.__version__)"  # may fail
```

**If `torch` is pip-installed** (e.g. `2.8.0+cu128`), install matching domain libs with **pip** — not micromamba:

```bash
# example for torch 2.8.0+cu128 → torchvision 0.23.0
micromamba remove torchvision torchaudio  # drop stale conda builds if present
pip install torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

See [pytorch.org previous versions](https://pytorch.org/get-started/previous-versions/) for other CUDA tags (`cu126`, `cu121`, `cpu`, …).

**Fresh conda-only install** (no pip torch):

```bash
micromamba env create -f environment_cuda.yml -p .env
```

**Checkpoint resume fails with `torch.load` / CVE-2025-32434**

Recent `transformers` requires `torch>=2.6` to load optimizer state. Upgrade the stack above, or delete `optimizer.pt` / `scheduler.pt` in the checkpoint folder and reload model weights only.

**WebDataset `curl` exit 56 during training**

Pre-download shards locally and point `--dataset_url` at local files instead of streaming from Hugging Face:

```bash
huggingface-cli download heesup/Cowpea-Architecture-XML-WDS --repo-type dataset --local-dir ./data/cowpea_wds
# then: --dataset_url "./data/cowpea_wds/shard-{000000..000200}.tar"
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
To train directly from the [Cowpea-Architecture-XML](https://huggingface.co/datasets/heesup/Cowpea-Architecture-XML) WebDataset shards:
```bash
python src/train_2.py \
    --dataset_url "https://huggingface.co/datasets/heesup/Cowpea-Architecture-XML-WDS/resolve/main/shard-{000000..000200}.tar" \
    --encoder_checkpoint facebook/dinov2-small \
    --decoder_checkpoint gpt2-medium \
    --image_size 448 \
    --batch_size 4 \
    --epoch 4
```

Shard splits default to `0-160` (train), `161-180` (val), and `181-200` (test). Override with `--train_shards`, `--val_shards`, and `--test_shards`.

### Inference
You can perform inference using the `PlantArchitectureModel` class. The following example demonstrates how to generate plant tokens from an image and convert them into a structured XML representation.

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
