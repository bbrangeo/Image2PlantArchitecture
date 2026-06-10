import argparse
import inspect
import os
import sys

# Required before torch import if --use_mps is passed (DINOv2 bicubic on MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, "..")
sys.path.insert(0, project_dir)
sys.path.insert(0, script_dir)

import numpy as np
import torch
from check_torch_stack import ensure_torchvision_compatible

ensure_torchvision_compatible()

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoImageProcessor, GPT2Config, Trainer, TrainingArguments

from cowpea_wds_dataset import (
    SHARD_RANGE_PATTERN,
    CowpeaWDSCollectedDataset,
    CowpeaWDSIterableDataset,
    build_shard_url,
    cache_wds_url,
    count_shards,
    is_remote_wds_url,
    parse_shard_range,
)
from models.model import PlantArchitectureModel
from plant_tokenizer import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, VOCAB_SIZE
from utils import model_summary


def _as_logits_tensor(output):
    """Keep only the [batch, seq, vocab] tensor from model outputs."""
    if output is None:
        return None
    if torch.is_tensor(output):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
        if output:
            return _as_logits_tensor(output[0])
    return output


class PlantArchitectureTrainer(Trainer):
    """Strip KV-cache objects before accelerate pads eval predictions."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
        )
        logits = _as_logits_tensor(logits)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def custom_data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])

    max_label_length = max(len(f["labels"]) for f in features)

    padded_labels = torch.stack([
        torch.cat([
            torch.tensor(f["labels"], dtype=torch.long),
            torch.full((max_label_length - len(f["labels"]),), PAD_TOKEN, dtype=torch.long),
        ])
        for f in features
    ])

    decoder_attention_mask = (padded_labels != PAD_TOKEN).long()
    labels = padded_labels.clone()
    labels[padded_labels == PAD_TOKEN] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
    }


def compute_metrics_for_training(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)

    valid_label_mask = (labels != -100) & (labels != PAD_TOKEN)
    valid_pred_mask = (predictions != -100) & (predictions != PAD_TOKEN)
    combined_mask = valid_label_mask & valid_pred_mask

    all_pred_tokens = predictions[combined_mask]
    all_label_tokens = labels[combined_mask]

    if len(all_pred_tokens) > 0 and len(all_label_tokens) > 0:
        try:
            micro_f1_score = f1_score(
                all_label_tokens, all_pred_tokens, average="weighted", zero_division=0
            )
            micro_accuracy_score = accuracy_score(all_label_tokens, all_pred_tokens)
        except Exception as exc:
            print(f"Error computing metrics: {exc}")
            micro_f1_score = 0.0
            micro_accuracy_score = 0.0
    else:
        micro_f1_score = 0.0
        micro_accuracy_score = 0.0

    return {
        "f1": micro_f1_score,
        "accuracy": micro_accuracy_score,
    }


def preprocess_logits_for_metrics(logits, labels):
    logits = _as_logits_tensor(logits)
    predictions = torch.argmax(logits, dim=-1)

    if predictions.shape[1] < labels.shape[1]:
        padding_size = labels.shape[1] - predictions.shape[1]
        padding = torch.full(
            (predictions.shape[0], padding_size),
            PAD_TOKEN,
            device=predictions.device,
            dtype=predictions.dtype,
        )
        predictions = torch.cat([predictions, padding], dim=1)
    elif predictions.shape[1] > labels.shape[1]:
        predictions = predictions[:, : labels.shape[1]]

    return predictions


def estimate_train_samples(total_samples: int, train_shards: str, total_shards: int) -> int:
    n_train_shards = count_shards(train_shards)
    return int(total_samples * n_train_shards / total_shards)


def get_device(use_mps: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if use_mps and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print(
            "Using MPS with CPU fallback (DINOv2 needs bicubic upsample, unsupported on MPS)."
        )
        return torch.device("mps")
    if torch.backends.mps.is_available():
        print(
            "MPS available but skipped: DINOv2 positional encoding uses bicubic "
            "upsample, which is not implemented on MPS. Using CPU."
        )
    return torch.device("cpu")


def supports_fp16(device: torch.device) -> bool:
    return device.type == "cuda"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Image to Plant Architecture model from a Hugging Face WebDataset"
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        required=True,
        help="WebDataset URL with braceexpand, e.g. .../shard-{000000..000039}.tar",
    )
    parser.add_argument("--train_shards", type=str, default="0-31")
    parser.add_argument("--val_shards", type=str, default="32-35")
    parser.add_argument("--test_shards", type=str, default="36-39")
    parser.add_argument("--total_samples", type=int, default=79560)
    parser.add_argument("--max_test_samples", type=int, default=500)
    parser.add_argument("--wds_workers", type=int, default=4)
    parser.add_argument(
        "--wds_cache_dir",
        type=str,
        default="data/cowpea_wds",
        help="Cache remote WebDataset shards locally (default: data/cowpea_wds). "
        "Set empty to stream over HTTP.",
    )
    parser.add_argument(
        "--stream_wds",
        type=str,
        default="False",
        help="Stream shards over HTTP instead of caching locally (not recommended on clusters)",
    )
    parser.add_argument("--image_size", type=int, default=448, help="Size of input images")
    parser.add_argument("--encoder_checkpoint", type=str, default="facebook/dinov2-small")
    parser.add_argument("--decoder_checkpoint", type=str, default="gpt2-medium")
    parser.add_argument(
        "--today_date_str",
        type=str,
        default="20250430_TrainValTestByShard",
        help="Date string for experiment naming",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="dinov2-small_448_WDS_gpt2-medium",
        help="Experiment name",
    )
    parser.add_argument("--curriculum", default="False", help="Not supported with WebDataset")
    parser.add_argument("--epoch", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--grad_acc", type=int, default=4, help="gradient_accumulation_steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (use 0 with WebDataset)")
    parser.add_argument("--color_jitter", type=str, default="False")
    parser.add_argument("--rnd_crop", type=str, default="False")
    parser.add_argument("--rnd_erase", type=str, default="False")
    parser.add_argument("--use_depth", type=str, default="False")
    parser.add_argument("--push_to_hub", type=str, default="True")
    parser.add_argument("--debug", type=str, default="False")
    parser.add_argument(
        "--use_mps",
        type=str,
        default="False",
        help="Use Apple MPS (slow fallback for DINOv2; CPU is default on Mac)",
    )

    args = parser.parse_args()

    args.curriculum = args.curriculum.lower() == "true"
    args.color_jitter = args.color_jitter.lower() == "true"
    args.rnd_crop = args.rnd_crop.lower() == "true"
    args.rnd_erase = args.rnd_erase.lower() == "true"
    args.use_depth = args.use_depth.lower() == "true"
    args.debug = args.debug.lower() == "true"
    args.push_to_hub = args.push_to_hub.lower() == "true"
    args.use_mps = args.use_mps.lower() == "true"
    args.stream_wds = args.stream_wds.lower() == "true"

    if args.curriculum:
        print("Warning: curriculum learning is not supported with WebDataset streaming and will be ignored.")

    exp_name = args.exp_name or "debug"
    output_base_dir = f"./log/{args.today_date_str}/{exp_name}" if args.today_date_str else f"./log/{exp_name}"
    os.makedirs(output_base_dir, exist_ok=True)
    results_dir = f"{output_base_dir}/results"

    decoder_checkpoint = args.decoder_checkpoint
    if "google-bert/bert" in decoder_checkpoint:
        decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 2500
        decoder_config.vocab_size = VOCAB_SIZE
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
    elif "gpt2" in decoder_checkpoint:
        decoder_config = GPT2Config.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 4096 * 2
        decoder_config.vocab_size = VOCAB_SIZE
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
    elif "google/bigbird-roberta" in decoder_checkpoint:
        decoder_config = AutoConfig.from_pretrained(decoder_checkpoint)
        decoder_config.max_position_embeddings = 4096 * 2
        decoder_config.vocab_size = VOCAB_SIZE
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        decoder_config.attention_type = "original_full"
    else:
        raise ValueError(f"Unsupported decoder checkpoint: {decoder_checkpoint}")

    decoder_config.decoder_start_token_id = SOS_TOKEN
    decoder_config.bos_token_id = SOS_TOKEN
    decoder_config.pad_token_id = PAD_TOKEN
    decoder_config.eos_token_id = EOS_TOKEN
    decoder_config.use_cache = False

    encoder_checkpoint = args.encoder_checkpoint
    image_size = args.image_size
    encoder_config = AutoConfig.from_pretrained(encoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    image_processor.crop_size["width"] = image_size
    image_processor.crop_size["height"] = image_size
    image_processor.size["shortest_edge"] = image_size

    n_gpu = max(1, torch.cuda.device_count())
    print(f"Available CUDA GPUs: {torch.cuda.device_count()}")

    if "RANK" in os.environ:
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        print(f"Initialized distributed training: rank={rank}, world_size={world_size}, device={device}")
    else:
        device = get_device(use_mps=args.use_mps)
        print(f"Using device: {device}")

    use_fp16 = supports_fp16(device)
    model_dtype = torch.float16 if use_fp16 else torch.float32

    model = PlantArchitectureModel.from_encoder_decoder_pretrained(
        encoder_checkpoint,
        decoder_checkpoint,
        decoder_config=decoder_config,
        encoder_config=encoder_config,
        decoder_ignore_mismatched_sizes=True,
        use_depth=args.use_depth,
        torch_dtype=model_dtype,
        tp_plan="auto",
    )
    model = model.to(device)
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.config.decoder_start_token_id = SOS_TOKEN
    model.config.bos_token_id = SOS_TOKEN
    model.config.pad_token_id = PAD_TOKEN
    model.config.eos_token_id = EOS_TOKEN
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    model.decoder.resize_token_embeddings(VOCAB_SIZE)

    torch.manual_seed(42)

    train_start, train_end = parse_shard_range(args.train_shards)
    val_start, val_end = parse_shard_range(args.val_shards)
    test_start, test_end = parse_shard_range(args.test_shards)

    shard_match = SHARD_RANGE_PATTERN.search(args.dataset_url)
    if shard_match is None:
        raise ValueError(f"Could not parse total shard count from: {args.dataset_url}")
    total_shards = int(shard_match.group(2)) - int(shard_match.group(1)) + 1

    train_url = build_shard_url(args.dataset_url, train_start, train_end)
    val_url = build_shard_url(args.dataset_url, val_start, val_end)
    test_url = build_shard_url(args.dataset_url, test_start, test_end)

    if args.wds_cache_dir and not args.stream_wds:
        for label, remote_url in (
            ("train", train_url),
            ("val", val_url),
            ("test", test_url),
        ):
            if is_remote_wds_url(remote_url):
                cached_url = cache_wds_url(remote_url, args.wds_cache_dir)
                if label == "train":
                    train_url = cached_url
                elif label == "val":
                    val_url = cached_url
                else:
                    test_url = cached_url

    print("Loading WebDataset...")
    print(f"  Train shards: {args.train_shards} -> {train_url}")
    print(f"  Val shards:   {args.val_shards} -> {val_url}")
    print(f"  Test shards:  {args.test_shards} -> {test_url}")

    train_samples = estimate_train_samples(args.total_samples, args.train_shards, total_shards)
    samples_per_epoch = train_samples // max(1, n_gpu)

    common_dataset_kwargs = {
        "image_processor": image_processor,
        "image_size": image_size,
        "process_leaf": True,
        "add_sos_token": False,
    }

    train_dataset = CowpeaWDSIterableDataset(
        url=train_url,
        mode="train",
        shardshuffle=True,
        workers=args.wds_workers,
        samples_per_epoch=samples_per_epoch,
        color_jitter=args.color_jitter,
        random_crop=args.rnd_crop,
        random_erase=args.rnd_erase,
        **common_dataset_kwargs,
    )
    val_dataset = CowpeaWDSIterableDataset(
        url=val_url,
        mode="val",
        shardshuffle=False,
        workers=max(1, args.wds_workers // 2),
        samples_per_epoch=None,
        **common_dataset_kwargs,
    )
    test_dataset = CowpeaWDSCollectedDataset(
        url=test_url,
        max_samples=args.max_test_samples if not args.debug else min(20, args.max_test_samples),
        mode="test",
        workers=max(1, args.wds_workers // 2),
        **common_dataset_kwargs,
    )

    batch_size = args.batch_size
    num_train_epochs = args.epoch
    gradient_accumulation_steps = args.grad_acc
    steps_per_epoch = max(
        1,
        train_samples // (batch_size * gradient_accumulation_steps * n_gpu),
    )
    max_steps = steps_per_epoch * num_train_epochs
    eval_steps = max(1, steps_per_epoch // 10)

    print(
        f"Estimated train samples: {train_samples}, "
        f"samples/epoch/gpu: {samples_per_epoch}, "
        f"steps/epoch: {steps_per_epoch}, max_steps: {max_steps}"
    )

    training_kwargs = dict(
        output_dir=f"{output_base_dir}/checkpoints",
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.2,
        weight_decay=0.01,
        logging_dir=f"{output_base_dir}/logs",
        logging_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=5,
        learning_rate=1e-4,
        dataloader_pin_memory=device.type == "cuda",
        dataloader_num_workers=args.num_workers,
        fp16=use_fp16,
        use_cpu=device.type == "cpu",
    )
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["save_safetensors"] = True
    training_args = TrainingArguments(**training_kwargs)

    trainer = PlantArchitectureTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics_for_training,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    model_summary(model=model, max_depth=1)

    if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
        print(f"Model checkpoint already exists at {results_dir}. Skipping training.")
        model = PlantArchitectureModel.from_pretrained(results_dir)
    else:
        print("Model training...")
        checkpoints_dir = os.path.join(output_base_dir, "checkpoints")
        if os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0:
            print("Model checkpoint already exists. Resuming training")
            trainer.train(resume_from_checkpoint=True)
        else:
            print("Training model from scratch")
            trainer.train()
        trainer.save_model(results_dir)

    if args.push_to_hub:
        import subprocess

        try:
            whoami_output = subprocess.check_output(["hf", "auth", "whoami"], text=True)
            username = whoami_output.strip().split("\n")[-1]
            if username.startswith("You are logged in as"):
                username = username.split("as")[-1].strip().split()[0]
        except Exception as exc:
            print(f"Could not determine Hugging Face username: {exc}")
            username = "your-username"
        repo_id = f"{username}/{exp_name}"
        print(f"Pushing model to Hugging Face Hub as {repo_id}")
        model.push_to_hub(repo_id)

    benchmark_folder = os.path.join(output_base_dir, "benchmark_results")
    benchmark_path = os.path.join(benchmark_folder, "benchmark.txt")
    if os.path.exists(benchmark_path):
        print("Benchmark already exists")
    else:
        print("Calculating metrics...")
        from calc_metric import calc_metric

        model.eval()
        calc_metric(
            model=model,
            test_dataset=test_dataset,
            log_path=benchmark_path,
            batch_size=batch_size,
            num_workers=args.num_workers,
            debug=args.debug,
            benchmark_folder=benchmark_folder,
        )
