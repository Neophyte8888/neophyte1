import cv2
import numpy as np
import os
import shutil
import random
import requests
import zipfile
import io
import tensorflow as tf
from model import create_simple_cnn

# --- Parameters ---
# Data directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")

# Data preparation
VALIDATION_SPLIT = 0.2

# Model training
MODEL_SAVE_PATH = 'models/aruco_defect_detector.keras'
IMG_SIZE = (200, 200)
BATCH_SIZE = 16
EPOCHS = 25 # Increased epochs for better training on real data

# Data URL
DATA_URL = 'https://drive.google.com/uc?export=download&id=1A2FuH4F2q6CBPhujIn-LQobxAbbRvTHM'


def download_and_unzip_data():
    """
    Downloads and unzips the data from the provided URL into the raw data directory.
    """
    print("--- Step 1: Downloading and Unzipping Data ---")

    # Ensure the raw data directory is clean and exists
    if os.path.exists(RAW_DATA_DIR):
        shutil.rmtree(RAW_DATA_DIR)
    os.makedirs(RAW_DATA_DIR)

    print(f"Downloading data from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all files, junking the paths to handle nested folders
            for member in z.infolist():
                if member.filename.endswith('.png'):
                    # Build a clean target path
                    target_path = os.path.join(RAW_DATA_DIR, os.path.basename(member.filename))
                    with open(target_path, 'wb') as f:
                        f.write(z.read(member.filename))

        print(f"Successfully downloaded and unzipped data to '{RAW_DATA_DIR}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        exit()
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        exit()

# --- Augmentation Functions ---
def add_gaussian_noise(image):
    row, col, ch = image.shape
    gauss = np.random.normal(0, 0.1**0.5, (row, col, ch))
    noisy = np.clip(image + (gauss * 50).reshape(row, col, ch), 0, 255)
    return noisy.astype(np.uint8)

def add_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def add_perspective_warp(image):
    rows, cols, _ = image.shape
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    offset = int(cols * 0.15)
    pts2 = np.float32([
        [random.randint(-offset, offset), random.randint(-offset, offset)],
        [cols-1-random.randint(-offset, offset), random.randint(-offset, offset)],
        [random.randint(-offset, offset), rows-1-random.randint(-offset, offset)],
        [cols-1-random.randint(-offset, offset), rows-1-random.randint(-offset, offset)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

def add_salt_and_pepper_noise(image, amount=0.01):
    """Adds salt and pepper noise to an image."""
    output = image.copy()
    # Salt mode
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[coords[0], coords[1], :] = (255,255,255)
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[coords[0], coords[1], :] = (0,0,0)
    return output

def add_occlusion(image, max_occlusions=3):
    """Adds random black shapes (rectangles or circles) to occlude parts of the image."""
    occluded_image = image.copy()
    height, width, _ = occluded_image.shape

    for _ in range(random.randint(1, max_occlusions)):
        # Choose between rectangle and circle
        if random.random() > 0.5:
            # Rectangle
            x1 = random.randint(0, width - 20)
            y1 = random.randint(0, height - 20)
            x2 = x1 + random.randint(20, 60) # Increased max size
            y2 = y1 + random.randint(20, 60)
            cv2.rectangle(occluded_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        else:
            # Circle
            cx = random.randint(20, width - 20)
            cy = random.randint(20, height - 20)
            radius = random.randint(10, 40)
            cv2.circle(occluded_image, (cx, cy), radius, (0, 0, 0), -1)

    return occluded_image

def augment_image_for_defect(image):
    """Applies a random, stronger set of augmentations."""
    augmentations = [
        add_gaussian_noise,
        add_salt_and_pepper_noise,
        add_blur,
        add_perspective_warp,
        add_occlusion
    ]
    # Apply more augmentations, from 2 to 4
    num_to_apply = random.randint(2, 4)
    # Ensure we don't try to apply more augmentations than available
    if num_to_apply > len(augmentations):
        num_to_apply = len(augmentations)

    applied_augmentations = random.sample(augmentations, num_to_apply)

    for aug_func in applied_augmentations:
        image = aug_func(image)
    return image

# --- Preparation Function ---
def prepare_dataset():
    """
    Creates 'ok' and 'defective' versions of images, splits them into
    training/validation sets, and saves them.
    """
    print(f"\n--- Step 2: Preparing and Augmenting Dataset ---")
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    os.makedirs(os.path.join(TRAIN_DIR, "ok"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, "defective"), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "ok"), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "defective"), exist_ok=True)
    print("Created processed data directories.")

    raw_images = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.png')]
    if not raw_images:
        print("Error: No images found in the raw data directory after download.")
        exit()

    random.shuffle(raw_images)
    split_idx = int(len(raw_images) * (1 - VALIDATION_SPLIT))
    train_files, validation_files = raw_images[:split_idx], raw_images[split_idx:]
    print(f"Splitting data: {len(train_files)} train, {len(validation_files)} validation.")

    def process_files(files, target_dir):
        for filename in files:
            img_path = os.path.join(RAW_DATA_DIR, filename)
            image = cv2.imread(img_path)
            if image is None: continue

            cv2.imwrite(os.path.join(target_dir, "ok", filename), image)
            defective_image = augment_image_for_defect(image)
            cv2.imwrite(os.path.join(target_dir, "defective", filename), defective_image)

    print("Processing training data...")
    process_files(train_files, TRAIN_DIR)
    print("Processing validation data...")
    process_files(validation_files, VALIDATION_DIR)
    print("Dataset preparation complete.")

# --- Training Function ---
def train_model():
    """
    Loads the dataset, creates the CNN model, trains it, and saves the result.
    """
    print(f"\n--- Step 3: Training the Model ---")

    print("Loading training data...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels='inferred', label_mode='binary', image_size=IMG_SIZE,
        interpolation='nearest', batch_size=BATCH_SIZE, shuffle=True
    )

    print("Loading validation data...")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR, labels='inferred', label_mode='binary', image_size=IMG_SIZE,
        interpolation='nearest', batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"Classes found: {train_dataset.class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    print("Creating the CNN model...")
    model = create_simple_cnn(input_shape=IMG_SIZE + (3,))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset
    )
    print("Model training finished.")

    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"\nFinal validation accuracy: {final_val_accuracy:.2%}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    return history

if __name__ == "__main__":
    download_and_unzip_data()
    prepare_dataset()
    train_model()
    print("\n--- All steps finished successfully! ---")
