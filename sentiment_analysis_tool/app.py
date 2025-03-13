import streamlit as st
import pandas as pd
from utils import load_models, analyze_data, generate_visualizations

# Initialize session state
if 'input_data' not in st.session_state:
    st.session_state.input_data = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Configure page
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main UI
def main():
    st.title("📊 Social Media Sentiment Analysis Tool")
    st.markdown("Analyze sentiment of social media content using multiple models")

    # Input Section
    with st.expander("📥 Input Data", expanded=True):
        input_method = st.radio("Select input method:", 
                              ["Text Input", "File Upload"], horizontal=True)
        
        if input_method == "Text Input":
            user_input = st.text_area(
                "Enter social media content (one entry per line):",
                "I love this product!\nThis is terrible...\nMeh, it's okay.",
                height=150
            )
            input_data = [line.strip() for line in user_input.split('\n') if line.strip()]
        else:
            uploaded_file = st.file_uploader("Upload CSV/Excel file", 
                                           type=["csv", "xlsx"])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    input_data = df.iloc[:, 0].astype(str).tolist()
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    input_data = []
            else:
                input_data = []

        st.session_state.input_data = input_data

    # Analysis Section
    st.divider()
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if st.session_state.input_data:
            with st.spinner("Analyzing sentiment..."):
                try:
                    models = load_models()
                    results = analyze_data(st.session_state.input_data, models)
                    st.session_state.analysis_results = results
                    st.success("Analysis completed!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please input some data first!")

    # Results Display
    if st.session_state.analysis_results:
        st.divider()
        generate_visualizations(st.session_state.analysis_results)

if __name__ == "__main__":
    main()
