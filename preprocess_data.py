import streamlit as st
import PyPDF2
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components
import json
from html import escape

# Path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "merged.csv")

# Load job data
jd_df = pd.read_csv(CSV_PATH)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# preprocess text
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\\s]', ' ', text)
    tokens = word_tokenize(text)
    processed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(processed)

def main():
    job_df = jd_df.copy()

    # Preprocess
    job_df['cleaned_description'] = job_df['Job Description'].fillna("").apply(preprocess_text)
    job_df.to_csv("preprocessed_data.csv",index=False)

if __name__ == '__main__':
    main()
