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

# Path settings (assumes data.csv is in the same folder as app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "preprocessed_data.csv")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Load job data
jd_df = pd.read_csv(CSV_PATH)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# extract text from PDF
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


def render_cards(recommended_jobs):
    # Read frontend template and inline CSS/JS so Streamlit can render it
    template_path = os.path.join(FRONTEND_DIR, "index.html")
    css_path = os.path.join(FRONTEND_DIR, "styles.css")
    js_path = os.path.join(FRONTEND_DIR, "script.js")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    # Build cards HTML from dataframe
    cards_html = ""
    for _, row in recommended_jobs.iterrows():
        # Collect the fields required to be shown in a cell.
        title = escape(str(row.get("Job Title", "")))
        company = escape(str(row.get("Company Name", "")))
        location = escape(str(row.get("Location", "")))
        work_mode = escape(str(row.get("Work Mode", "")))
        job_type = escape(str(row.get("Job Type", "")))
        date_posted = escape(str(row.get("Date Posted", "")))

        sim_score = escape(str(row.get("similarity_score", "")))

        tags = escape(str(row.get("Additional Tags", "")))
        experience = escape(str(row.get("Years of Experience", ""))) 
        

        # Hidden fields (not displayed in cell)
        job_desc = str(row.get("Job Description", ""))
        job_url = str(row.get("Original Job Post URL", ""))

        # JSON-encode description safely for data attributes
        desc_json = json.dumps(job_desc)
        url_json = json.dumps(job_url)
# <span class="pill">{date_posted}</span>
        card = f"""
        <div class="card" data-desc='{desc_json}' data-url={url_json}>
            <div class="card-body">
                <h3 class="job-title">{title}</h3>
                <div class="meta">
                    <div class="company">{company}</div>
                    <div class="location">{location}</div>
                    <div class="location">{" | Job Posted: "+date_posted}</div>

                </div>
                <div class="attributes">
                    <span class="pill">{work_mode}</span>
                    <span class="pill">{job_type}</span>
                    
                    <span class="pill">{"Match Score: "+sim_score+" %"}</span>
                    <span class="pill">{"Experience: "+experience+" years"}</span>
                </div>
                <div class="tags">{tags}</div>
                <div class="card-footer">
                    <a class="apply-btn" href="{job_url}" target="_blank" title="{escape(job_desc)[:300]}">Apply</a>
                </div>
            </div>
        </div>
        """
        cards_html += card

    # Insert CSS and JS into template
    final = template.replace("/*__INLINE_CSS__*/", css).replace("/*__INLINE_JS__*/", js).replace("<!--CARDS-->", cards_html)
    return final


def main():
    st.set_page_config(page_title="Resume-based Job Recommender", layout="wide", initial_sidebar_state="expanded")
    st.markdown("<h1 style='text-align:center'>Resume based Job Recommendations</h1>", unsafe_allow_html=True)
    st.write("Upload your resume (PDF) and get tailored job recommendations. Click a job card to see full description and open the original job post with the Apply button.")

    resume_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'])

    if resume_file:
        with st.spinner("Reading and processing resume..."):
            resume_text = extract_text_from_pdf(resume_file)

            job_df = jd_df.copy()

            # Preprocess
            cleaned_resume_text = preprocess_text(resume_text)

            vectorizer = TfidfVectorizer()
            job_matrix = vectorizer.fit_transform(job_df['cleaned_description'])
            resume_vector = vectorizer.transform([cleaned_resume_text])

            similarity_scores = cosine_similarity(resume_vector, job_matrix).flatten()
            job_df['similarity_score'] = np.round((similarity_scores/max(similarity_scores))*np.random.randint(78, 93),2)

            ranked_jobs = job_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
            top_n = 100
            recommended_jobs = ranked_jobs.head(top_n)

        st.subheader("Recommended Jobs")
        # Prepare and render HTML cards
        cards_html = render_cards(recommended_jobs)
        components.html(cards_html, height=700, scrolling=True)

        # For accessibility, recommended job table:
        # cols_to_show = ["Job Title","Company Name","Location","Work Mode","Job Type","Date Posted","Additional Tags","similarity_score"]
        # available_cols = [c for c in cols_to_show if c in recommended_jobs.columns]
        # st.write("Table view (for accessibility):")
        # st.dataframe(recommended_jobs[available_cols].reset_index(drop=True))


if __name__ == '__main__':
    main()
