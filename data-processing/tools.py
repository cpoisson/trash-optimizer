import os
import random
from PIL import Image
from torchvision import transforms

def balance_dataset(dataset_path):
    """
    Équilibre toutes les classes d'un dataset

    On augmente chaque classe jusqu'à atteindre le nombre d'images
    de la classe la plus grande.
    """

    # Transformation simple et robuste
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # 1) Lire les classes
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])

    # 2) Compter les images par classe
    class_counts = {}
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        class_counts[cls] = len(imgs)

    # 3) Trouver la classe max
    target_size = max(class_counts.values())
    print("Taille cible pour chaque classe :", target_size)

    # 4) Augmenter chaque classe
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        current = len(images)

        print(f"\nClasse {cls} : {current} → {target_size}")

        # Rien à faire si déjà suffisant
        if current >= target_size:
            print("Déjà suffisante.")
            continue

        i = 0
        while current < target_size:
            # choisir image source existante
            img_name = random.choice(images)
            img_path = os.path.join(class_path, img_name)

            img = Image.open(img_path).convert("RGB")

            # appliquer augmentation
            aug_tensor = augment(img)
            aug_img = transforms.ToPILImage()(aug_tensor)

            # nouveau nom
            save_path = os.path.join(class_path, f"aug_{i}.jpg")
            aug_img.save(save_path)

            current += 1
            i += 1

        print(f"✔ Classe {cls} équilibrée ({current} images).")
