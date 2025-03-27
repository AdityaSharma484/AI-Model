import os
import cv2
import pytesseract
import pandas as pd
import shutil
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pdf2image import convert_from_path

# Load trained model and encoder
model_path = "/content/expense_validation_model.keras"
model = tf.keras.models.load_model(model_path)

# Function to preprocess image for prediction
def preprocess_image_for_model(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize pixels
    return np.expand_dims(img, axis=0)

# Function to classify expense type
def classify_expense(image_path):
    processed_image = preprocess_image_for_model(image_path)
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_label, confidence

# Function to extract text from images
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return ""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    text = pytesseract.image_to_string(gray)
    return text

# Function to extract details from text
def extract_bill_details(text):
    bill_number = None
    date = None
    company_name = None
    
    bill_pattern = re.search(r'INVOICE\s*No[:\s]*([A-Za-z0-9-]+)', text, re.IGNORECASE)
    date_patterns = [r'(\d{1,2}-[A-Za-z]+-\d{2,4})', r'(\d{1,2}/\d{1,2}/\d{2,4})', r'(\d{1,2}\.\d{1,2}\.\d{2,4})']
    company_pattern = re.search(r'(TRUE SALES PHARMA EXPENSE VALIDATION)', text, re.IGNORECASE)

    if bill_pattern:
        bill_number = bill_pattern.group(1)
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date = match.group(1)
            break
    
    if company_pattern:
        company_name = company_pattern.group(1)

    return bill_number, date, company_name

# Process multiple bills with classification
def process_bills(folder_path):
    bills = []
    existing_bills = set()
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print("Processing file:", file_path)
        
        text = ""
        if filename.endswith(".pdf"):
            try:
                images = convert_from_path(file_path)
                if images:
                    temp_image_path = "temp_image.jpg"
                    images[0].save(temp_image_path, 'JPEG')
                    text = extract_text_from_image(temp_image_path)
                    file_path = temp_image_path  # Use image path for classification
                else:
                    print(f"Error: No pages found in {filename}")
            except Exception as e:
                print(f"Error processing PDF {filename}: {e}")
                continue
        else:
            text = extract_text_from_image(file_path)
        
        bill_no, date, company_name = extract_bill_details(text)
        
        duplicate_flag = "No"
        if not bill_no or not date or not company_name:
            duplicate_flag = "Yes"
        elif (bill_no and bill_no in existing_bills) or (date and date in existing_bills):
            duplicate_flag = "Yes"
        elif company_name and company_name.upper() != "TRUE SALES PHARMA EXPENSE VALIDATION":
            duplicate_flag = "Yes"
        else:
            if bill_no:
                existing_bills.add(bill_no)
            if date:
                existing_bills.add(date)

        # Classify the image using the model
        predicted_label, confidence = classify_expense(file_path)

        bills.append({
            "Filename": filename,
            "Bill No": bill_no if bill_no else "Not Detected",
            "Date": date if date else "Not Detected",
            "Flagged": duplicate_flag,
            "Predicted Class": predicted_label,
            "Confidence Score": round(confidence, 2),
        })
    
    return pd.DataFrame(bills)

# Run the pipeline
df = process_bills("/content/expense_dataset/image_filename")

# Display extracted bill details
print("\n=== Extracted Bill Data with Classification ===")
print(df)