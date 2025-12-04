import os
import random
from PIL import Image
from torchvision import transforms

def balance_dataset(dataset_path):
    """
    Balance all classes in a dataset. Each class will have the same number of images.
    """

    # Simple and robust transformation pipeline
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # 1) Read all class directories
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])

    # 2) Count images per class
    class_counts = {}
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        class_counts[cls] = len(imgs)

    # 3) Find the maximum class size
    target_size = max(class_counts.values())
    print("Target size for each class:", target_size)

    # 4) Augment each class to match target size
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        current = len(images)

        print(f"\nClass {cls}: {current} → {target_size}")

        # Skip if already at target size
        if current >= target_size:
            print("Already sufficient.")
            continue

        i = 0
        while current < target_size:
            # Select a random existing source image
            img_name = random.choice(images)
            img_path = os.path.join(class_path, img_name)

            img = Image.open(img_path).convert("RGB")

            # Apply augmentation transformations
            aug_tensor = augment(img)
            aug_img = transforms.ToPILImage()(aug_tensor)

            # Generate new filename for augmented image
            save_path = os.path.join(class_path, f"aug_{i}.jpg")
            aug_img.save(save_path)

            current += 1
            i += 1

        print(f"✔ Class {cls} balanced ({current} images).")

def main():
    pass

if __name__ == "__main__":
    main()
