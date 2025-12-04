import csv
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

def load_mapping(csv_path: Path):
    """Load CSV mapping file (source_class,target_class) and return as dictionary."""
    mapping = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)  # Creates a reader that transforms each row into a dictionary with column names as keys
        for row in reader:
            mapping[row["source_class"].strip()] = row["target_class"].strip()
    return mapping

def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS

def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def assemble_one_dataset(dataset_dir: Path, mapping: dict, output_dir: Path):
    """Copy images while remapping them to target classes."""
    counts = {}
    ignored = []

    for source_class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        source_class = source_class_dir.name
        if source_class not in mapping:
            ignored.append(source_class)
            continue
        target_class = mapping[source_class]
        for img_path in source_class_dir.rglob("*"):
            if not img_path.is_file() or not is_image(img_path):
                continue
            if not is_valid_image(img_path):
                continue
            dest_dir = output_dir / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_name = f"{dataset_dir.name}_{img_path.name}"
            shutil.copy2(img_path, dest_dir / new_name)
            counts[target_class] = counts.get(target_class, 0) + 1
    return counts, ignored

def assemble_all(root_processing: Path, root_processed: Path):
    """Iterate through all datasets under root_processing/datasets and assemble them."""
    all_counts = {}
    all_ignored = {}
    datasets_dir = root_processing / "datasets"
    mappings_dir = root_processing / "mappings"

    for dataset_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
        mapping_file = mappings_dir / f"{dataset_dir.name}.csv"
        if not mapping_file.exists():
            # Fallback to lowercase naming to tolerate case mismatches between folders and CSVs
            fallback = mappings_dir / f"{dataset_dir.name.lower()}.csv"

            if fallback.exists():
                mapping_file = fallback
        if not mapping_file.exists():
            all_ignored[dataset_dir.name] = ["<no mapping file>"]  # No assembly possible
            continue
        mapping = load_mapping(mapping_file)
        counts, ignored = assemble_one_dataset(dataset_dir, mapping, root_processed)
        all_counts[dataset_dir.name] = counts
        if ignored:
            all_ignored[dataset_dir.name] = ignored

    return all_counts, all_ignored

def list_class_images(root_processed: Path):
    """Return a dictionary mapping target classes to lists of image paths from ROOT_PROCESSED."""
    class_to_images = {}
    for class_dir in sorted(p for p in root_processed.iterdir() if p.is_dir()):
        images = [
            img for img in sorted(class_dir.rglob("*"))
            if img.is_file() and is_image(img)
        ]
        if images:
            class_to_images[class_dir.name] = images
    return class_to_images

def augment_image(src: Path, dest: Path, rng: random.Random):
    """Create an augmented image (flip/slight rotation/color jitter) and save it."""
    with Image.open(src) as img:
        choice = rng.choice(["flip", "rot", "jitter"])
        if choice == "flip":
            aug = ImageOps.mirror(img)
        elif choice == "rot":
            angle = rng.uniform(-12, 12)  # Small rotation
            aug = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        else:
            # Light jitter on brightness and contrast
            bright = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
            aug = ImageEnhance.Contrast(bright).enhance(rng.uniform(0.9, 1.1))

        fmt = img.format  # Preserve original format if known
        aug.save(dest, format=fmt)

def balance_dataset(root_processed: Path, output_balanced: Path, target: int | None = 1000, seed: int = 42):
    """
    Rebalance classes by oversampling (with augmentation) minority classes.

    Args:
        root_processed: Directory containing images already remapped by target class.
        output_balanced: Output directory for balanced dataset.
        target: Number of images per class (default 1000, or minimum if None).
        seed: Random seed for reproducible selection of images to augment.
    """
    rng = random.Random(seed)
    class_to_images = list_class_images(root_processed)
    if not class_to_images:
        return {}, {}

    counts = {cls: len(imgs) for cls, imgs in class_to_images.items()}
    target = target if target is not None else min(counts.values())

    output_counts = {}
    generated = {}
    output_balanced.mkdir(parents=True, exist_ok=True)

    for cls, images in class_to_images.items():
        dest_dir = output_balanced / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        # If the class already has enough images, randomly select "target" images (reproducibly)
        selected = sorted(images)
        if len(selected) >= target:
            chosen = rng.sample(selected, target) if len(selected) > target else selected
            for img_path in chosen:
                shutil.copy2(img_path, dest_dir / img_path.name)
            output_counts[cls] = target
            continue

        # Otherwise, copy all existing images and fill the deficit with augmented images until reaching "target"
        for img_path in selected:
            shutil.copy2(img_path, dest_dir / img_path.name)

        deficit = target - len(selected)
        generated[cls] = []
        if selected:
            for i in range(deficit):
                base = rng.choice(selected)
                new_name = f"{base.stem}_aug{i}{base.suffix}"
                aug_path = dest_dir / new_name
                augment_image(base, aug_path, rng)
                generated[cls].append(aug_path)

        output_counts[cls] = target

    return output_counts, generated
