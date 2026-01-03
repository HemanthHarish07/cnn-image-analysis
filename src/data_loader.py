import os
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

def collect_image_paths(dataset_path):
    image_paths = []
    labels = []

    # Iterate through class folders and collect valid image files
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_dir):
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        for file in files:
            image_paths.append(os.path.join(class_dir, file))
            labels.append(class_name)

    return image_paths, labels

def stratified_split(image_paths, labels, seed=42):
    # Split into 70% training and 30% temporary (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed
    )

    # Split temporary set into 50% validation and 50% test (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test