import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def screen_resume(job_description, resume_text):
    content = [job_description, resume_text]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(count_matrix)
    match_percentage = round(similarity_matrix[0][1] * 100, 2)
    
    jd_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    matched_skills = [word for word in jd_words.intersection(resume_words) if len(word) > 3]
    
    return match_percentage, matched_skills

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("📄 AI-Based Resume Screening System")
st.subheader("Automating Recruitment with NLP")

st.divider()

# Inputs
jd_input = st.text_area("Paste Job Description Here:", height=150)
resume_input = st.text_area("Paste Candidate Resume Here:", height=250)

if st.button("Analyze Resume"):
    if jd_input and resume_input:
        score, skills = screen_resume(jd_input, resume_input)
        
        # Results Display
        st.write("### Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Match Score", value=f"{score}%")
        
        with col2:
            status = "SELECTED" if score >= 40 else "REJECTED"
            st.info(f"Decision: {status}")

        st.write("**Matched Keywords:**")
        st.write(", ".join(skills) if skills else "No major keywords matched.")
        
        # Visual Progress Bar
        st.progress(score / 100)
    else:
        st.error("Please provide both a Job Description and a Resume.")


        