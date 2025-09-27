import os
import random
import shutil
import json
from collections import defaultdict
from pathlib import Path
import time
import numpy as np


def validate_acdc_structure(root_path):
    """Validate the structure of the ACDC dataset (for .nii files)"""
    root_path = Path(root_path)
    training_dir = root_path / "training"

    if not training_dir.exists():
        raise ValueError(f"Training directory not found: {training_dir}")

    valid_patients = 0
    patient_sizes = {}

    for patient_dir in training_dir.iterdir():
        if not patient_dir.is_dir() or not patient_dir.name.startswith("patient"):
            continue

        frame_files = list(patient_dir.glob("*_frame??.nii"))
        gt_files = list(patient_dir.glob("*_frame??_gt.nii"))

        if len(frame_files) >= 2 and len(gt_files) >= 2:
            valid_patients += 1
            patient_sizes[patient_dir.name] = len(frame_files)
        else:
            print(f"Invalid patient {patient_dir.name}: frames({len(frame_files)}) labels({len(gt_files)})")

    print(f"\nValidation complete: Found {valid_patients} valid patients")
    return valid_patients > 0, patient_sizes


def load_acdc_samples(root_path):
    """Load ACDC samples (for .nii files)"""
    root_path = Path(root_path)
    patient_data = defaultdict(list)

    for patient_dir in (root_path / "training").iterdir():
        if not patient_dir.is_dir() or not patient_dir.name.startswith("patient"):
            continue

        for frame_file in patient_dir.glob("*_frame??.nii"):
            if "_gt" in frame_file.name:
                continue

            gt_file = patient_dir / f"{frame_file.stem}_gt.nii"

            if gt_file.exists():
                patient_data[patient_dir.name].append({
                    'patient_id': patient_dir.name,
                    'image_path': str(frame_file.absolute()),
                    'label_path': str(gt_file.absolute()),
                    'frame_num': frame_file.stem.split("_frame")[-1]
                })

    return patient_data


def copy_to_split_dir(samples, target_dir):
    """Copy samples to the target directory"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        patient_dir = target_dir / sample['patient_id']
        patient_dir.mkdir(exist_ok=True)

        shutil.copy2(sample['image_path'], patient_dir)
        shutil.copy2(sample['label_path'], patient_dir)


def save_split_info(output_dir, split_samples, split_name, seed):
    """Save split information to a JSON file"""
    split_info = {
        'split_name': split_name,
        'seed': seed,
        'patient_count': len({s['patient_id'] for s in split_samples}),
        'sample_count': len(split_samples),
        'samples': [
            {
                'patient_id': s['patient_id'],
                'frame_num': s['frame_num'],
                'image_path': s['image_path'],
                'label_path': s['label_path']
            } for s in split_samples
        ]
    }

    os.makedirs(output_dir / "splits", exist_ok=True)
    with open(output_dir / "splits" / f"{split_name}_seed{seed}.json", "w") as f:
        json.dump(split_info, f, indent=2)


def split_acdc_dataset(root_path, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """Main function for random splitting of the ACDC dataset"""
    random.seed(seed)
    np.random.seed(seed)

    print(f"\nUsing fixed seed: {seed} to split the dataset")

    print("Validating dataset structure...")
    valid, patient_sizes = validate_acdc_structure(root_path)
    if not valid:
        raise ValueError("Dataset structure validation failed")

    print("\nLoading data samples...")
    patient_data = load_acdc_samples(root_path)
    total_patients = len(patient_data)
    total_samples = sum(len(samples) for samples in patient_data.values())

    if total_patients == 0:
        raise ValueError("No valid data samples loaded")

    print(f"Total patients: {total_patients}, total samples: {total_samples}")

    patients = sorted(patient_data.keys())
    random.shuffle(patients)

    train_end = int(total_patients * train_ratio)
    val_end = train_end + int(total_patients * val_ratio)

    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]

    train_samples = [s for p in train_patients for s in patient_data[p]]
    val_samples = [s for p in val_patients for s in patient_data[p]]
    test_samples = [s for p in test_patients for s in patient_data[p]]

    output_root = Path(root_path)

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    test_dir = output_root / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)

    print("\nCopying files to target directories...")
    copy_to_split_dir(train_samples, train_dir)
    copy_to_split_dir(val_samples, val_dir)
    copy_to_split_dir(test_samples, test_dir)

    save_split_info(output_root, train_samples, "train", seed)
    save_split_info(output_root, val_samples, "val", seed)
    save_split_info(output_root, test_samples, "test", seed)

    print("\n" + "=" * 60)
    print(f"{'ACDC Dataset Split Results':^60}")
    print("=" * 60)
    print(f"Train set: {len(train_patients)} patients ({len(train_samples)} samples)")
    print(f"Validation set: {len(val_patients)} patients ({len(val_samples)} samples)")
    print(f"Test set: {len(test_patients)} patients ({len(test_samples)} samples)")
    print(f"Random seed: {seed}")
    print("=" * 60)

    print(f"\nSplit info saved to: {output_root / 'splits'}")
    print("For training, please use the following parameter:")
    print(f"--split_seed {seed}")

    return seed


def load_split(root_path, split_name, seed=None):
    """Load a specific split"""
    splits_dir = Path(root_path) / "splits"
    split_file = splits_dir / f"{split_name}_seed{seed}.json"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        split_info = json.load(f)

    print(f"Loaded {split_info['split_name']} split (seed: {split_info['seed']})")
    print(f"Number of patients: {split_info['patient_count']}, number of samples: {split_info['sample_count']}")
    return split_info['samples']


if __name__ == "__main__":

    DATA_ROOT = Path("./data/acdc")
    SEED = 42  # Set to None to generate a random seed

    try:
        used_seed = split_acdc_dataset(DATA_ROOT, seed=SEED)

        try:
            train_samples = load_split(DATA_ROOT, "train")
            print(f"Successfully loaded {len(train_samples)} training samples")
        except Exception as e:
            print(f"Failed to load training set: {str(e)}")

    except Exception as e:
        print("\n" + "!" * 60)
        print(f"{' Error Occurred ':^60}")
        print("!" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("\nDebug suggestions:")
        print(f"1. Make sure the path exists: {DATA_ROOT}")
        print("!" * 60)