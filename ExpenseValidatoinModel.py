import os
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import cv2
import pytesseract
from PIL import Image
import unidecode
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def unzip_dataset(zip_path='/content/data.zip', extract_path='/content/expense_dataset'):
    """
    Unzip the dataset with specific file handling
    """
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"✅ Dataset extracted to {extract_path}")
    print("Contents of extraction folder:", os.listdir(extract_path))
    
    return extract_path

def load_dataset(csv_path='/content/expense_dataset/csv_data.csv', image_folder='/content/expense_dataset'):
    """
    Load CSV and match with image files
    """
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded CSV with {len(df)} rows")
    print("CSV Columns:", list(df.columns))
    
    if 'filename' not in df.columns:
        raise KeyError("❌ The column 'filename' is missing in the CSV. Please check the file format.")
    
    if not os.path.isdir(image_folder):
        subfolders = [f for f in os.listdir('/content/expense_dataset') if os.path.isdir(os.path.join('/content/expense_dataset', f))]
        if subfolders:
            image_folder = os.path.join('/content/expense_dataset', subfolders[0])
    
    df['image_path'] = df['filename'].apply(
        lambda x: os.path.join(image_folder, x) if os.path.exists(os.path.join(image_folder, x)) else None
    )
    
    df_cleaned = df.dropna(subset=['image_path'])
    print(f"🖼️ Valid images found: {len(df_cleaned)} out of {len(df)} original rows")
    
    return df_cleaned

def load_images(image_paths, target_size=(128, 128)):
    """
    Loads images from file paths and preprocesses them into numpy arrays.
    """
    images = []
    for path in image_paths:
        img = load_img(path, target_size=target_size)  # Load image
        img = img_to_array(img) / 255.0  # Normalize pixels
        images.append(img)
    return np.array(images)

def train_expense_validation_model(df):
    """
    Train a CNN model for expense validation.
    """
    X = load_images(df['image_path'])
    y = df['class']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, label_encoder

def expense_validation_workflow(zip_path='/content/data.zip'):
    """
    Complete workflow from ZIP to trained model
    """
    extraction_path = unzip_dataset(zip_path)
    df = load_dataset()
    model, label_encoder = train_expense_validation_model(df)
    
    model.save('/content/expense_validation_model.keras')
    print("🎉 Model Training Complete!")
    print("Model saved to: /content/expense_validation_model.keras")
    
    return model, label_encoder

if _name_ == '_main_':
    try:
        model, label_encoder = expense_validation_workflow()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Troubleshooting steps:")
        print("1. Verify your ZIP file is uploaded")
        print("2. Check that 'csv_data.csv' is the correct filename")
        print("3. Ensure images are in a subfolder within the ZIP")

print("Expense Validation AI Model is ready to process your dataset!")