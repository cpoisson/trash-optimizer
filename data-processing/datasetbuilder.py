import csv
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
import toml
import time
import os
from dotenv import load_dotenv
load_dotenv()
CONFIGURATION_FILE = os.getenv("BUILDER_CONFIGURATION", "config.toml")

class DataSetBuilderConfig:
    """Configuration for dataset building and processing."""

    def __init__(self, config_path: Path):
        config = self.load_configuration(config_path)["configuration"]
        self.input_root_dir = Path(config["input_root_dir"]).expanduser()
        self.input_img_extensions = config.get("input_img_extensions", [".jpg", ".jpeg", ".png"])
        self.output_root_dir = Path(config["output_root_dir"]).expanduser()
        self.output_dataset_name = config["output_dataset_name"]
        self.output_max_per_category = config["output_max_per_category"]
        self.output_minimum_images_size_wh = tuple(config["output_minimum_images_size_wh"])
        self.output_categories = config.get("output_categories", [])
        self.datasets = config.get("datasets", [])

    @staticmethod
    def load_configuration(config_path: Path):
        """Load toml configuration file"""
        # Verify configuration file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return toml.load(config_path)

class ImageDataset:
    def __init__(self, name: str, path: Path, config: DataSetBuilderConfig, input_output_categories: dict):
        self.name = name
        self.path = path
        self.config = config
        self.input_output_categories = input_output_categories
        self.input_image_categories_paths = {}
        self.output_image_categories_paths = {}

    def __add__(self, other):
        """Combine two ImageDataset instances into one."""
        combined = ImageDataset(
            name=f"{self.name}_{other.name}",
            path=self.path,  # Path is not particularly relevant for combined dataset
            config=self.config,
            input_output_categories={**self.input_output_categories, **other.input_output_categories}
        )

        # Combine input image categories paths
        combined.input_image_categories_paths = {
            **self.input_image_categories_paths
        }
        for category, paths in other.input_image_categories_paths.items():
            if category in combined.input_image_categories_paths:
                combined.input_image_categories_paths[category].extend(paths)
            else:
                combined.input_image_categories_paths[category] = paths

        # Combine output image categories paths (contains tagged images)
        combined.output_image_categories_paths = {
            **self.output_image_categories_paths
        }
        for category, tagged_paths in other.output_image_categories_paths.items():
            if category in combined.output_image_categories_paths:
                combined.output_image_categories_paths[category].extend(tagged_paths)
            else:
                combined.output_image_categories_paths[category] = list(tagged_paths)

        # Shuffle combined paths to ensure fair mixing from both datasets
        for category in combined.output_image_categories_paths:
            random.shuffle(combined.output_image_categories_paths[category])

        return combined

    def if_valid_image(self, img_path: Path) -> bool:
        """Check if image is valid and supported."""
        if img_path.suffix.lower() not in self.config.input_img_extensions:
            return False
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify that it is, in fact an image
                if (
                    img.width < self.config.output_minimum_images_size_wh[0]
                    or img.height < self.config.output_minimum_images_size_wh[1]
                ):
                    return False
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def is_valid_input_category(self, category: str) -> bool:
        """Check if category is in the input categories"""
        if category not in self.input_output_categories.keys():
            return False
        return True

    def _find_category_folders(self, directory: Path, max_depth: int, current_depth: int = 0) -> dict:
        """Recursively search for folders matching category names.

        Args:
            directory: Directory to search in
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            Dictionary mapping category names to their folder paths
        """
        category_folders = {}

        if current_depth > max_depth:
            return category_folders

        try:
            for item in sorted(directory.iterdir()):
                if not item.is_dir():
                    continue

                # Check if this folder name matches a category
                if self.is_valid_input_category(item.name):
                    category_folders[item.name] = item

                # Continue searching in subdirectories
                if current_depth < max_depth:
                    sub_categories = self._find_category_folders(item, max_depth, current_depth + 1)
                    category_folders.update(sub_categories)
        except (PermissionError, OSError):
            pass

        return category_folders

    def _collect_images_recursive(self, directory: Path, images: list, current_depth: int, max_depth: int):
        """Recursively collect valid images from directory up to max_depth.

        Args:
            directory: Current directory to search
            images: List to append found images to
            current_depth: Current recursion depth
            max_depth: Maximum allowed recursion depth
        """
        if current_depth > max_depth:
            return

        for item in sorted(directory.iterdir()):
            if item.is_file() and self.if_valid_image(item):
                images.append(item)
            elif item.is_dir() and current_depth < max_depth:
                self._collect_images_recursive(item, images, current_depth + 1, max_depth)

    def load(self, max_depth: int = 3):
        """Load images from dataset path. Verify images are valid and supported.
        Searches recursively for category folders matching input_output_categories.

        Args:
            max_depth: Maximum depth to search for category folders (default: 3)
        """

        # Search for category folders recursively
        category_folders = self._find_category_folders(self.path, max_depth=max_depth)

        for class_name, class_dir in category_folders.items():
            # Collect images up to 2 levels deep within each category folder
            images = []
            self._collect_images_recursive(class_dir, images, current_depth=0, max_depth=2)

            if images:
                self.input_image_categories_paths[class_name] = images
                print(f"  Loaded {len(images)} images for category '{class_name}'")

        # Fill output categories paths based on input-output mapping
        # Store as tuples: (image_path, dataset_name) for tracking contribution
        for input_cat, images in self.input_image_categories_paths.items():
            output_cat = self.input_output_categories.get(input_cat)
            if output_cat:
                if output_cat not in self.output_image_categories_paths:
                    self.output_image_categories_paths[output_cat] = []
                # Tag each image with dataset source
                tagged_images = [(img, self.name) for img in images]
                self.output_image_categories_paths[output_cat].extend(tagged_images)


    def save(self, output_path: Path):
        """Save images to output path according to input-output category mapping.
        Supports many-to-one mapping where multiple input categories map to same output.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Save using output_image_categories_paths which contains (path, dataset_name) tuples
        for output_category, tagged_images in self.output_image_categories_paths.items():
            # Create output category directory
            output_category_dir = output_path / output_category
            output_category_dir.mkdir(parents=True, exist_ok=True)

            # Copy images with dataset source clearly visible in filename
            for img_data in tagged_images:
                # Handle both tuple format (img_path, dataset_name) and plain path
                if isinstance(img_data, tuple):
                    img_path, dataset_source = img_data
                else:
                    img_path = img_data
                    dataset_source = self.name

                # Format: {dataset_source}_{original_filename}
                new_name = f"{dataset_source}_{img_path.name}"
                dest_path = output_category_dir / new_name

                # Handle duplicate filenames by adding counter
                counter = 1
                while dest_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    new_name = f"{dataset_source}_{stem}_{counter}{suffix}"
                    dest_path = output_category_dir / new_name
                    counter += 1

                shutil.copy2(img_path, dest_path)

        print(f"Dataset saved to {output_path}")



class DataSetBuilder:
    """Class to build and process datasets based on configuration."""

    def __init__(self, config: DataSetBuilderConfig):
        self.config = config
        self.datasets = [
            ImageDataset(
                name=ds["folder_name"],
                path=Path(config.input_root_dir).expanduser() / Path(ds["folder_name"]),
                config=config,
                input_output_categories=ds.get("input_output_categories", {})
            )
            for ds in config.datasets
        ]
        self.output_dataset = None

    def create_dataset(self):
        """ Assemble datasets according to configuration.
        """
        print("Creating output dataset...")

        # Load all datasets
        for dataset in self.datasets:
            print(f"Loading dataset: {dataset.name}")
            dataset.load()

        # Combine all datasets using the __add__ operator
        if not self.datasets:
            raise ValueError("No datasets configured")

        self.output_dataset = self.datasets[0]
        for dataset in self.datasets[1:]:
            print(f"Combining dataset: {dataset.name}")
            self.output_dataset = self.output_dataset + dataset

        print(f"✓ Combined {len(self.datasets)} datasets into output dataset")

    def balance_datasets(self, seed: int = 42):
        """ Balance datasets using augmentation to reach target sizes"""
        print("Balancing datasets...")
        if not self.output_dataset:
            raise ValueError("No output dataset to balance. Run create_dataset first.")

        rng = random.Random(seed)
        target_size = self.config.output_max_per_category

        # Get category counts
        category_counts = {}
        for input_cat, images in self.output_dataset.input_image_categories_paths.items():
            output_cat = self.output_dataset.input_output_categories.get(input_cat, input_cat)
            category_counts[output_cat] = category_counts.get(output_cat, 0) + len(images)

        print(f"Current category distribution: {category_counts}")
        print(f"Target size per category: {target_size}")

        # Balance each output category
        for output_cat, tagged_images in self.output_dataset.output_image_categories_paths.items():
            current_count = len(tagged_images)

            if current_count >= target_size:
                continue  # Already balanced

            deficit = target_size - current_count
            print(f"Augmenting category '{output_cat}' with {deficit} images.")

            augmented_images = []
            i = 0
            while deficit > 0:
                # Choose random image from existing ones
                img_data = rng.choice(tagged_images)
                if isinstance(img_data, tuple):
                    base_img_path, dataset_source = img_data
                else:
                    base_img_path = img_data
                    dataset_source = "unknown"

                with Image.open(base_img_path) as img:
                    # Convert to RGB to avoid palette mode issues when saving as JPEG
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Simple augmentation: flip or rotate
                    if rng.choice([True, False]):
                        aug_img = ImageOps.mirror(img)
                    else:
                        angle = rng.uniform(-15, 15)
                        aug_img = img.rotate(angle, expand=True)

                    # Save augmented image to temporary path
                    aug_img_name = f"{dataset_source}_aug_{i}_{base_img_path.stem}.jpg"
                    aug_img_path = base_img_path.parent / aug_img_name
                    aug_img.save(aug_img_path, 'JPEG')
                    # Tag augmented image with source dataset
                    augmented_images.append((aug_img_path, dataset_source))

                deficit -= 1
                i += 1

            # Add augmented images to output dataset
            self.output_dataset.output_image_categories_paths[output_cat].extend(augmented_images)

    def save_dataset(self):
        """ Save the processed dataset to output directory """
        print("Saving dataset...")
        if not self.output_dataset:
            raise ValueError("No output dataset to save. Run create_dataset first.")
        # Add timestamp to output path to avoid overwriting
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_folder_name = f"{self.config.output_dataset_name}_{timestamp}"
        output_path = self.config.output_root_dir / output_folder_name
        self.output_dataset.save(output_path)
        # Add a detailed summary file showing per-dataset contributions
        summary_path = output_path / "dataset_summary.txt"
        with open(summary_path, "w") as summary_file:
            summary_file.write("Dataset Summary\n")
            summary_file.write("=================\n\n")

            total_images = 0
            dataset_contributions = {}

            for category, tagged_images in self.output_dataset.output_image_categories_paths.items():
                summary_file.write(f"Category: {category}\n")
                summary_file.write(f"Total images: {len(tagged_images)}\n")

                # Count contribution per source dataset
                source_counts = {}
                for img_data in tagged_images:
                    if isinstance(img_data, tuple):
                        _, dataset_source = img_data
                    else:
                        dataset_source = "unknown"
                    source_counts[dataset_source] = source_counts.get(dataset_source, 0) + 1
                    dataset_contributions[dataset_source] = dataset_contributions.get(dataset_source, 0) + 1

                # Show breakdown by source
                summary_file.write("  Contribution by dataset:\n")
                for source, count in sorted(source_counts.items()):
                    percentage = (count / len(tagged_images)) * 100
                    summary_file.write(f"    - {source}: {count} ({percentage:.1f}%)\n")
                summary_file.write("\n")

                total_images += len(tagged_images)

            # Overall summary
            summary_file.write("\n" + "="*50 + "\n")
            summary_file.write("Overall Dataset Contribution:\n")
            summary_file.write(f"Total images: {total_images}\n\n")
            for source, count in sorted(dataset_contributions.items()):
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                summary_file.write(f"  - {source}: {count} images ({percentage:.1f}%)\n")

        print(f"✓ Dataset saved to: {output_path}")
        print(f"\n📊 Dataset contribution summary:")
        for source, count in sorted(dataset_contributions.items()):
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  - {source}: {count} images ({percentage:.1f}%)")


def main():
    config = DataSetBuilderConfig(Path(CONFIGURATION_FILE))
    builder = DataSetBuilder(config)
    builder.create_dataset()
    builder.balance_datasets()
    builder.save_dataset()



if __name__ == "__main__":
    main()
