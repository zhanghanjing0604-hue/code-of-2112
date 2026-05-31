import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


# These are the image file types this script will accept.
# If the dataset contains a file with another extension, it will be ignored.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The Kaggle dataset uses folder/file names related to "healthy" and "bleached".
# This dictionary tells the script which words should count as each class.
# Example: if a folder is called "bleached_corals", it belongs to the "bleached" class.
CLASS_ALIASES = {
    "healthy": ("healthy", "unbleached", "normal"),
    "bleached": ("bleached", "bleach"),
}


def parse_args():
    # argparse lets us change settings from the terminal if needed.
    # For example:
    # python prepare_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
    # If we do not type extra options, the default values below are used.
    parser = argparse.ArgumentParser(
        description="Split the Kaggle healthy/bleached coral dataset into train, val, and test folders."
    )
    parser.add_argument(
        "--raw-dir",
        default="raw_data",
        help="Folder containing the downloaded and unzipped Kaggle dataset.",
    )
    parser.add_argument("--output-dir", default="data", help="Output folder for train/val/test data.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them. Copy is safer and is the default behavior in this script.",
    )
    return parser.parse_args()


def find_images_for_class(raw_dir, class_name):
    # This function searches inside raw_data/ and finds images for one class.
    # class_name is either "healthy" or "bleached".
    aliases = CLASS_ALIASES[class_name]
    matched_images = []

    # rglob("*") means: look through this folder and all subfolders.
    for path in raw_dir.rglob("*"):
        # Skip folders and skip files that are not images.
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Convert folder names and file names to lowercase so matching is easier.
        # Example: "Bleached Corals" becomes "bleached_corals".
        parts = [part.lower().replace(" ", "_") for part in path.parts]
        filename = path.stem.lower().replace(" ", "_")

        # If either the file name or the folder path contains a class keyword,
        # keep this image for that class.
        if any(alias in filename for alias in aliases) or any(
            any(alias in part for alias in aliases) for part in parts
        ):
            matched_images.append(path)

    return sorted(set(matched_images))


def clear_split_dirs(output_dir):
    # Before creating a new train/val/test split, remove old copied images.
    # This prevents mixing an old split with a new split.
    # .gitkeep files are kept because they only exist to keep empty folders visible.
    for split in ("train", "val", "test"):
        for class_name in CLASS_ALIASES:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_path in class_dir.iterdir():
                if image_path.is_file() and image_path.name != ".gitkeep":
                    image_path.unlink()


def copy_split(files, destination_dir):
    # Copy a list of image files into one destination folder.
    # The number at the front of the file name keeps the output ordered and unique.
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files, start=1):
        target = destination_dir / f"{index:04d}_{source.name}"
        shutil.copy2(source, target)


def split_files(files, train_ratio, val_ratio, test_ratio, seed):
    # Machine learning datasets are commonly split into:
    # train: used to teach the model
    # val: used during training to check if the model is improving
    # test: used at the end as the final evaluation
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")

    # First split: keep 70% for training, and put 30% into a temporary group.
    train_files, temp_files = train_test_split(
        files,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Second split: divide that temporary group into validation and test sets.
    # With the defaults, this makes 15% validation and 15% test overall.
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_fraction_of_temp,
        random_state=seed,
        shuffle=True,
    )
    return train_files, val_files, test_files


def main():
    # main() is the starting point of this script.
    # It reads the settings, finds the raw images, splits them, and copies them.
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw_dir}")

    clear_split_dirs(output_dir)

    # Do the same process once for "healthy" and once for "bleached".
    for class_name in CLASS_ALIASES:
        files = find_images_for_class(raw_dir, class_name)
        if not files:
            raise FileNotFoundError(
                f"No images found for class '{class_name}' in {raw_dir}. "
                "Check the unzipped Kaggle folder names."
            )

        # Randomly split this class into train/val/test.
        # The seed makes the split repeatable, so the result is the same each time.
        train_files, val_files, test_files = split_files(
            files,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )

        # Copy the images into the folders used by TensorFlow.
        copy_split(train_files, output_dir / "train" / class_name)
        copy_split(val_files, output_dir / "val" / class_name)
        copy_split(test_files, output_dir / "test" / class_name)

        print(
            f"{class_name}: {len(train_files)} train, "
            f"{len(val_files)} val, {len(test_files)} test"
        )

    print("\nDataset preparation complete.")
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()
