import streamlit as st
import numpy as np
from utils.file_parser import parse_resume
from utils.text_processing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    st.title("🤖 AI Resume Screener")
    
    # User inputs
    job_description = st.text_area("📝 Paste Job Description", height=200)
    resumes = st.file_uploader("📁 Upload Resumes", 
                             type=["pdf", "docx"], 
                             accept_multiple_files=True)
    
    if st.button("🚀 Screen Resumes") and job_description and resumes:
        with st.spinner("🔍 Analyzing documents..."):
            # Process job description
            processed_jd = preprocess_text(job_description)
            
            # Process resumes
            resume_data = []
            for resume in resumes:
                try:
                    text = parse_resume(resume)
                    processed_text = preprocess_text(text)
                    resume_data.append({
                        "name": resume.name,
                        "text": processed_text,
                        "raw_text": text[:500] + "..."  # For preview
                    })
                except Exception as e:
                    st.error(f"❌ Error processing {resume.name}: {str(e)}")
            
            if not resume_data:
                st.error("⚠️ No valid resumes processed")
                return

            # Create TF-IDF matrix
            corpus = [processed_jd] + [r["text"] for r in resume_data]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Calculate similarity scores
            jd_vector = tfidf_matrix[0]
            resume_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(jd_vector, resume_vectors)[0]
            
            # Rank resumes
            ranked_indices = np.argsort(similarities)[::-1]
            
            # Display results
            st.subheader("📊 Ranking Results")
            cols = st.columns([1, 3, 2])
            cols[0].markdown("**Rank**")
            cols[1].markdown("**Resume**")
            cols[2].markdown("**Score**")
            
            for idx, rank_idx in enumerate(ranked_indices):
                resume = resume_data[rank_idx]
                cols = st.columns([1, 3, 2])
                cols[0].markdown(f"`#{idx+1}`")
                cols[1].expander(resume["name"]).write(resume["raw_text"])
                cols[2].progress(similarities[rank_idx])
                cols[2].markdown(f"`{similarities[rank_idx]:.2%}`")

if __name__ == "__main__":
    main()
