import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Plagiarism Checker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for theming
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #2E3B55 0%, #1C2533 100%);
        color: #EAEAEA;
    }
    /* Main header style */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 0;
    }
    /* Sub-header */
    .sub-header {
        text-align: center;
        color: #B0C4DE;
    }
    /* File uploader container */
    .file-uploader {
        background: #33415C;
        padding: 1rem;
        border-radius: 8px;
    }
    /* Results table */
    .dataframe tbody tr th {
        color: #ff0000;
    }
    .dataframe tbody tr td {
        color: #ff0000;
    }
    /* Buttons */
    .stButton>button {
        background-color: #FFD700;
        color: #1C2533;
        font-weight: bold;
    }
    /* Slider */
    .stSlider>div div input {
        accent-color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown("<h1 class='main-header'>📚 Assignment Plagiarism Checker</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload assignments and detect similarities with ease.</p>", unsafe_allow_html=True)

st.write("Developed by:")
st.write("Nagumothu Pavan Kumar")
st.write("SRM University, Ramapuram")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙ Settings")
    threshold = st.slider("Similarity Threshold", 0, 100, 75)
    st.markdown("---")

# File Upload Section
st.markdown("### 📂 Upload Documents", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Select .txt files", type=["txt"], accept_multiple_files=True, key="uploader")

if uploaded_files and len(uploaded_files) > 1:
    st.success(f"{len(uploaded_files)} assignments loaded.")

    # Read content
    texts = [f.read().decode("utf-8") for f in uploaded_files]
    names = [f.name.replace('.txt','') for f in uploaded_files]

    # Compute TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(texts)
    cos_sim = cosine_similarity(matrix)

    # Build results
    data = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            score = cos_sim[i][j]
            if score >= threshold:
                data.append({'Student 1': names[i],
                             'Student 2': names[j],
                             'Similarity Score': round(score, 2)})

    st.markdown("### 🧾 Results", unsafe_allow_html=True)
    if data:
        df = pd.DataFrame(data)
        # Color scale based on score for Similarity Score column
        def color_score(val):
            if val > 0.9:
                color = '#F08080'  # Light coral for high similarity (>0.9)
            elif val > 0.8:
                color = '#FF9999'  # Lighter coral for medium similarity (>0.8)
            else:
                color = '#FFB6C1'  # Very light coral/pink for lower similarity
            return f'background-color: {color}; color: #1C2533;'  # Dark blue text for contrast

        # Apply styling only to the Similarity Score column
        styled = df.style.applymap(color_score, subset=['Similarity Score'])
        st.write(styled)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name='plagiarism_report.csv',
            mime='text/csv'
        )
    else:
        st.info("✅ No pairs exceed the threshold.")

elif uploaded_files:
    st.warning("Please upload at least two files to analyze.")
else:
    st.info("Waiting for assignment uploads...")
