# Expense Validation AI - README

## **Project Overview**

Expense validation is a critical process in financial management to verify the authenticity of invoices and prevent fraud. This project leverages **AI, OCR (Optical Character Recognition), and Deep Learning** to automate invoice classification, extract relevant information, and flag potential duplicates.

## **Features**

- **Automated Invoice Classification:** Uses a CNN model to classify documents.
- **Text Extraction via OCR:** Utilizes Tesseract OCR to extract invoice details.
- **Duplicate & Fraud Detection:** Identifies duplicate invoices and validates company authenticity.
- **Pipeline for Processing:** Accepts PDFs and images, extracts text, and processes the data.

## **Technologies Used**

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow, Keras
- **OCR Library:** Tesseract OCR
- **Image Processing:** OpenCV
- **Data Handling:** Pandas, NumPy
- **PDF Processing:** pdf2image

## **Installation & Setup**

### **1. Install Dependencies**

Run the following commands to install the necessary packages:

```bash
pip install tensorflow opencv-python pytesseract pandas numpy pdf2image
sudo apt-get install tesseract-ocr poppler-utils
```

### **2. Run the Application**

```bash
python main.py
```

## **How It Works**

1. **Upload invoices** (PDFs or images) into the system.
2. **OCR extracts text** from the document.
3. **Regex-based extraction** identifies key details:
   - Invoice Number
   - Date
   - Company Name
4. **Classification model** determines document type.
5. **Flagging system** checks for duplicate or fraudulent invoices.
6. **Results** are displayed and stored in a structured format.

## **Model Training**

- CNN model trained on **expense datasets**.
- Images are resized to **128x128** for classification.
- Training uses **80% data**, validation uses **20%**.
- Softmax activation for **multi-class classification**.

## **Challenges & Solutions**

| **Challenge**                      | **Solution**                                           |
| ---------------------------------- | ------------------------------------------------------ |
| OCR accuracy on low-quality images | Applied grayscale conversion and adaptive thresholding |
| Handling different invoice formats | Used regex-based pattern extraction                    |
| Duplicate detection                | Implemented cross-checking with stored data            |
