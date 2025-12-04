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

        # Combine output image categories paths
        combined.output_image_categories_paths = {
            **self.output_image_categories_paths
        }
        for category, paths in other.output_image_categories_paths.items():
            if category in combined.output_image_categories_paths:
                combined.output_image_categories_paths[category].extend(paths)
            else:
                combined.output_image_categories_paths[category] = paths

        # shuffle combined paths to mix images from both datasets
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

    def load(self, max_depth: int = 2):
        """Load images from dataset path. Verify images are valid and supported.

        Args:
            max_depth: Maximum depth to search for images in subdirectories (default: 2)
        """

        for class_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            class_name = class_dir.name
            if not self.is_valid_input_category(class_name):
                print(f"Skipping invalid category: {class_name}")
                continue

            # Collect images up to max_depth levels deep
            images = []
            self._collect_images_recursive(class_dir, images, current_depth=0, max_depth=max_depth)

            if images:
                self.input_image_categories_paths[class_name] = images
                print(f"  Loaded {len(images)} images for category '{class_name}'")

        # Fill output categories paths based on input-output mapping
        for input_cat, images in self.input_image_categories_paths.items():
            output_cat = self.input_output_categories.get(input_cat)
            if output_cat:
                if output_cat not in self.output_image_categories_paths:
                    self.output_image_categories_paths[output_cat] = []
                self.output_image_categories_paths[output_cat].extend(images)


    def save(self, output_path: Path):
        """Save images to output path according to input-output category mapping.
        Supports many-to-one mapping where multiple input categories map to same output.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        for input_category, images in self.input_image_categories_paths.items():
            output_category = self.input_output_categories.get(input_category)
            if not output_category:
                print(f"Warning: No output mapping for input category '{input_category}' in dataset '{self.name}'")
                continue

            # Create output category directory
            output_category_dir = output_path / output_category
            output_category_dir.mkdir(parents=True, exist_ok=True)

            # Copy images with unique names to avoid conflicts
            for img_path in images:
                # Include dataset name and input category in filename to ensure uniqueness
                new_name = f"{self.name}_{input_category}_{img_path.name}"
                dest_path = output_category_dir / new_name
                shutil.copy2(img_path, dest_path)

        print(f"Dataset '{self.name}' saved to {output_path}")



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

        # Combine datasets
        self.output_dataset = self.datasets[0]


        print(f"Combined {len(self.datasets)} datasets into output dataset")

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

        # Balance each category
        for input_cat, images in self.output_dataset.input_image_categories_paths.items():
            output_cat = self.output_dataset.input_output_categories.get(input_cat, input_cat)
            current_count = category_counts.get(output_cat, 0)

            if current_count >= target_size:
                continue  # Already balanced

            deficit = target_size - current_count
            print(f"Augmenting category '{output_cat}' with {deficit} images.")

            augmented_images = []
            i = 0
            while deficit > 0:
                base_img_path = rng.choice(images)
                with Image.open(base_img_path) as img:
                    # Simple augmentation: flip or rotate
                    if rng.choice([True, False]):
                        aug_img = ImageOps.mirror(img)
                    else:
                        angle = rng.uniform(-15, 15)
                        aug_img = img.rotate(angle, expand=True)

                    # Save augmented image to temporary path
                    aug_img_name = f"aug_{i}_{base_img_path.name}"
                    aug_img_path = base_img_path.parent / aug_img_name
                    aug_img.save(aug_img_path)
                    augmented_images.append(aug_img_path)

                deficit -= 1
                i += 1

            # Add augmented images to dataset
            self.output_dataset.input_image_categories_paths[input_cat].extend(augmented_images)

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
        # Add a summary file
        summary_path = output_path / "dataset_summary.txt"
        with open(summary_path, "w") as summary_file:
            summary_file.write("Dataset Summary\n")
            summary_file.write("=================\n\n")
            for category, images in self.output_dataset.output_image_categories_paths.items():
                summary_file.write(f"Category: {category}\n")
                summary_file.write(f"Number of images: {len(images)}\n\n")
        print(f"✓ Dataset saved to: {output_path}")


def main():
    config = DataSetBuilderConfig(Path(CONFIGURATION_FILE))
    builder = DataSetBuilder(config)
    builder.create_dataset()
    builder.balance_datasets()
    builder.save_dataset()



if __name__ == "__main__":
    main()
