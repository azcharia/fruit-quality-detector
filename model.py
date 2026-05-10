import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import joblib
import numpy as np
import shutil
import cv2
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
IMAGE_SIZE_CNN = 224
IMAGE_SIZE_ML = 64
BATCH_SIZE = 32
MODEL_NAME_BASE = "fruit_classifier_model"
ML_MODEL_NAME = "fruit_classifier_ml.joblib"
CLASS_NAMES_FILE = "class_names.joblib"
DATASET_DIR = "dataset/train"

def create_cnn_model(num_classes: int):
    """Creates a MobileNetV2 based model for transfer learning."""
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE_CNN, IMAGE_SIZE_CNN, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_cnn(validation_split: float):
    """Trains the CNN model."""
    train_ratio = int((1 - validation_split) * 100)
    val_ratio = int(validation_split * 100)
    split_name = f"{train_ratio}_{val_ratio}"
    
    print(f"\n--- TRAINING CNN (Split {train_ratio}-{val_ratio}) ---")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=(IMAGE_SIZE_CNN, IMAGE_SIZE_CNN),
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=(IMAGE_SIZE_CNN, IMAGE_SIZE_CNN),
        batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    joblib.dump(class_names, CLASS_NAMES_FILE)
    
    normalization_layer = layers.Rescaling(1./127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    
    model = create_cnn_model(len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=5)
    
    model_filename = f"{MODEL_NAME_BASE}_{split_name}.keras"
    model.save(model_filename)
    
    return history.history['val_accuracy'][-1], model_filename

def train_ml(validation_split: float):
    """Trains a Random Forest classifier as the ML comparison."""
    print(f"\n--- TRAINING ML (Random Forest, Split {int((1-validation_split)*100)}-{int(validation_split*100)}) ---")
    
    images = []
    labels = []
    class_names = sorted(os.listdir(DATASET_DIR))
    
    # Feature extraction: Resize to 64x64, Grayscale, Flatten
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path): continue
        
        print(f"Processing class: {class_name}")
        for img_name in tqdm(os.listdir(class_path)[:500]): # Limit to 500 images per class for ML speed
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE_ML, IMAGE_SIZE_ML))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img.flatten())
                labels.append(label_idx)
    
    X = np.array(images)
    y = np.array(labels)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"ML Accuracy: {acc:.4f}")
    
    joblib.dump(rf, ML_MODEL_NAME)
    return acc

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    # 1. Train ML (Random Forest) - Using 85-15 split for ML
    ml_acc = train_ml(0.15)
    
    # 2. Train CNN with two splits
    splits = [0.15, 0.25]
    cnn_results = []

    for val_split in splits:
        acc, filename = train_cnn(val_split)
        cnn_results.append({"split": f"{int((1-val_split)*100)}-{int(val_split*100)}", "accuracy": acc, "file": filename})

    print("\n\n" + "#"*50)
    print(" FINAL COMPARISON SUMMARY ")
    print("#"*50)
    print(f"Machine Learning (Random Forest): Accuracy = {ml_acc:.4f}")
    for res in cnn_results:
        print(f"CNN Split {res['split']}: Accuracy = {res['accuracy']:.4f}")
    
    # Default CNN model for app
    best_cnn = max(cnn_results, key=lambda x: x['accuracy'])
    shutil.copy(best_cnn['file'], f"{MODEL_NAME_BASE}.keras")
    print(f"\nBest CNN model ({best_cnn['split']}) set as default.")

if __name__ == "__main__":
    main()
