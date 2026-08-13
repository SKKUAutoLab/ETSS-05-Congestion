
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_CUSTOM_DATA_DIRS = ("data/sta", "custom_data", "data/custom", "data/custom_data")
NUMBER_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _number_dir(target_dir, number):
    for name in (str(number), NUMBER_WORDS[number - 2]):
        candidate = target_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _existing_data_root(data_dir=None):
    roots = [Path(data_dir)] if data_dir else [Path(path) for path in DEFAULT_CUSTOM_DATA_DIRS]
    root = next((path for path in roots if path.is_dir()), None)
    if root is None:
        expected = ", ".join(str(path) for path in roots)
        raise FileNotFoundError(
            f"Custom dataset directory not found. Expected one of: {expected}. "
            "Use --data_dir to point to your dataset."
        )
    return root, roots


def is_sta_dataset(root):
    root = Path(root)
    return (
        (root / "train").is_dir()
        and any((root / "train").glob("*.jpg"))
        and any((root / "train").glob("*_dmap.npy"))
    )


def _density_count(dmap_path):
    return float(np.load(dmap_path).sum())


def _sta_samples(split_dir, thresholds):
    samples = []
    for image_path in sorted(split_dir.glob("*.jpg")):
        dmap_path = image_path.with_name(f"{image_path.stem}_dmap.npy")
        if not dmap_path.is_file():
            continue
        count = _density_count(dmap_path)
        label = int(np.searchsorted(thresholds, count, side="right"))
        samples.append(
            {
                "image_path": str(image_path),
                "dmap_path": str(dmap_path),
                "count": count,
                "label": label,
            }
        )
    return samples


def load_sta_dataset(data_dir=None, split="test", num_classes=4):
    root, _ = _existing_data_root(data_dir)
    if not is_sta_dataset(root):
        raise ValueError(f"{root} does not look like an STA crowd-counting dataset.")

    train_counts = [
        _density_count(dmap_path)
        for dmap_path in sorted((root / "train").glob("*_dmap.npy"))
    ]
    if not train_counts:
        raise ValueError(f"No training density maps found under {root / 'train'}.")

    split_dir = root / split
    if not split_dir.is_dir():
        raise ValueError(f"Split directory not found: {split_dir}")

    quantiles = np.linspace(0, 1, num_classes + 1)[1:-1]
    thresholds = np.quantile(np.array(train_counts, dtype=np.float32), quantiles)
    samples = _sta_samples(split_dir, thresholds)
    if not samples:
        raise ValueError(f"No STA image/density-map pairs found under {split_dir}.")

    return {
        "root": str(root),
        "split": split,
        "thresholds": thresholds.tolist(),
        "samples": samples,
    }


def load_custom_dataset(data_dir=None, num_classes=4):
    root, roots = _existing_data_root(data_dir)
    if is_sta_dataset(root):
        raise ValueError(
            f"{root} is an STA crowd-counting dataset, not a folder-per-count custom dataset."
        )

    augmented_data = {}
    required_numbers = range(2, num_classes + 2)
    for target_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target_data = {}
        for number in required_numbers:
            number_dir = _number_dir(target_dir, number)
            if number_dir is None:
                continue
            samples = []
            for image_path in sorted(number_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                with Image.open(image_path) as image:
                    samples.append({"img": image.convert("RGB").copy(), "path": str(image_path)})
            if samples:
                target_data[number] = samples
        if all(number in target_data for number in required_numbers):
            augmented_data[target_dir.name] = target_data

    if not augmented_data:
        expected = ", ".join(str(path) for path in roots)
        raise ValueError(
            f"No complete custom dataset found under {root}. Searched: {expected}. "
            "Expected folders like "
            "<root>/<object>/2/*.jpg, <root>/<object>/3/*.jpg, <root>/<object>/4/*.jpg, "
            "and <root>/<object>/5/*.jpg."
        )

    sample_size = min(
        len(augmented_data[target][number])
        for target in augmented_data
        for number in required_numbers
    )
    return augmented_data, sample_size

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="An example script to parse command-line arguments.")

    # Add arguments
    parser.add_argument("-o","--dataset",type=str,choices=["custom","countbench"],help="choose from custom dataset or countbench")
    parser.add_argument("-t","--task",type=str,choices=["classification","image_retrievel","image_gen"],help="choose the task")
    parser.add_argument("-m","--model",type=str,choices=["clip_base_32","clip_base_16","clip_large_14","stable_diffusion"],help="choose the task")
    parser.add_argument("-r","--ref_obj",type=str,help="name of the object being used as an reference")
    parser.add_argument("--data_dir",type=str,default=None,help="path to custom dataset root")
    parser.add_argument("--sample_size",type=int,default=None,help="number of images per class to evaluate")
    parser.add_argument("--split",type=str,default="test",choices=["train","val","test"],help="STA split to evaluate")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_directory = "reproduced_results"
    Path(local_directory).mkdir(parents=True, exist_ok=True)

    # run image generation with stable diffucsion
    if args.task == "image_gen" and args.model == "stable_diffusion":
        from sd import reproduce_stable_diffusion_results

        pretrained_model_name="CompVis/stable-diffusion-v1-4"
        reproduce_stable_diffusion_results(local_directory,pretrained_model_name,device)
    elif args.model in ["clip_base_32","clip_base_16","clip_large_14"]:
        from clip import image_retrievel, img_clf, img_clf_sta

        if args.dataset=="custom":
            root, _ = _existing_data_root(args.data_dir)
            if is_sta_dataset(root):
                if args.task != "classification":
                    raise ValueError("STA crowd-counting data is only supported for --task classification.")
                crowd_data = load_sta_dataset(args.data_dir, split=args.split)
                img_clf_sta(
                    args.model,
                    args.ref_obj or "crowd",
                    crowd_data,
                    local_directory,
                    device=device,
                    sample_size=args.sample_size,
                )
            elif args.task == "classification":
                augmented_data, available_sample_size = load_custom_dataset(args.data_dir)
                sample_size = args.sample_size or available_sample_size
                if sample_size > available_sample_size:
                    raise ValueError(
                        f"--sample_size {sample_size} is larger than the smallest class size "
                        f"({available_sample_size}) in the custom dataset."
                    )
                img_clf(args.model,args.ref_obj,sample_size,augmented_data,local_directory,device=device)
            elif args.task == "image_retrievel":
                augmented_data, available_sample_size = load_custom_dataset(args.data_dir)
                sample_size = args.sample_size or available_sample_size
                if sample_size > available_sample_size:
                    raise ValueError(
                        f"--sample_size {sample_size} is larger than the smallest class size "
                        f"({available_sample_size}) in the custom dataset."
                    )
                image_retrievel(args.model,args.ref_obj,sample_size,augmented_data,local_directory,device=device)
        elif (args.dataset=="countbench") and (args.task == "classification"):
            # TODO: load dataset
            # TODO: run on countbench
            countbench_dat = None
