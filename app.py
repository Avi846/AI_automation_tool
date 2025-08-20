import os
import openai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# File processing function
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# AI processing functions
def ai_task(user_input, task_type, target_language = None):
    try: # Inside this, that code is written which can give error
        if task_type == "Summarize":
            prompt = f"Summarize this text:\n{user_input}"
            system_message = "You are an AI assistant that summarizes text."
        elif task_type == "Translate":
            prompt = f"Translate this text into {target_language}:\n{user_input}"
            system_message = f"You are a professional translator that translates text into {target_language}."
        elif task_type == "Rephrase":
            prompt = f"Rephrase this text professionally:\n{user_input}"
            system_message = "You are an AI text assistant."
        elif task_type == "Key Points":
            prompt = f"Extract key bullet points from this text:\n{user_input}"
            system_message = "You are an AI text assistant."
        elif task_type == "Summary Report":
            prompt = f"Create a professional summary report from this content:\n\n{user_input}"
            system_message = "You are a professional report writer."
        elif task_type == "Professional Email":
            prompt = f"Draft a professional email based on the following content:\n\n{user_input}"
            system_message = "You are a professional email writer."
        elif task_type == "Dataset Analysis":
            prompt = f"""You are a data analyst. Analyze the following dataset and create a 
            structured report with:
            - Key Insights
            - Trend and Patterns
            - Anomalies (if any)
            - Recommendations
            
            Dataset:\n\n{user_input}"""
            system_message = "You are a senior data analyst with 10+ years of experience."

        response = openai.ChatCompletion.create(
            model = "openai/gpt-3.5-turbo",
            messages = [
                {"role":"system", "content": system_message},
                {"role":"user", "content": prompt}
            ],
            max_tokens = 800,
            temperature = 0.5
        )
        return str(response.choices[0].message["content"])
    
    except Exception as e: # If code inside try give error, then program jump here and take alter action, if there is no error, then except code is skiped
        st.error(f"Error in AI processing: {str(e)}")
        return None
    
# Data Visualization functions
def show_data_visualizations(df):
    st.subheader("Data Visualizations")

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include = ["int64","float64"]).columns
    if len(numeric_cols) > 0:
        st.markdown("### Numeric Data Analysis")
        col1, col2 = st.columns(2)

        with col1:
            selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num_col], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_num_col], ax=ax)
            st.pyplot(fig)
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        st.markdown("### Categorical Data Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=df[selected_cat_col], 
                     order=df[selected_cat_col].value_counts().iloc[:15].index,
                     ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) > 1:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ========== Streamlit UI ==========
st.set_page_config(page_title="AI Super Analyzer", layout="wide")

st.title(" AI Super Analyzer")
st.write("Combine text processing, document analysis, and data visualization in one powerful tool.")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Text Processing", "Document Analysis", "Dataset Explorer"])

with tab1:
    st.header(" Text Processing")
    st.write("Perform text transformations like summarization, translation, and more.")
    
    sub_tab1, sub_tab2 = st.tabs(["Basic Mode", "Advanced Mode"])
    
    with sub_tab1:
        task_type = st.selectbox("Choose a task:", ["Summarize", "Translate", "Rephrase", "Key Points"])
        
        # Show language input only when translation is selected
        target_language = None
        if task_type == "Translate":
            target_language = st.text_input(" Enter target language (e.g., French, Spanish, Japanese):")
        
        user_text = st.text_area(" Enter your text here:", height=150, key="basic_text")
        
        if st.button("Run AI", key="basic_button"):
            if user_text.strip() == "":
                st.warning(" Please enter some text.")
            elif task_type == "Translate" and not target_language.strip():
                st.warning(" Please enter a target language.")
            else:
                with st.spinner("Processing... "):
                    output = ai_task(user_text, task_type, target_language)
                
                if output:
                    st.success(" Task Completed!")
                    st.write(output)
                    st.download_button(
                        label="⬇ Download Output",
                        data=output,
                        file_name="text_output.txt",
                        mime="text/plain"
                    )
    
    with sub_tab2:
        user_text = st.text_area(" Enter your text here:", height=150, key="advanced_text")
        target_language = st.text_input(" Enter target language for translation (leave blank for no translation):")
        
        if st.button("Generate Output", key="advanced_button"):
            if user_text.strip() == "":
                st.warning(" Please enter some text.")
            else:
                with st.spinner("Generating summary..."):
                    ai_summary = ai_task(user_text, "Summarize")
                
                if ai_summary:
                    st.subheader(" AI Summary")
                    st.write(ai_summary)
                    
                    if target_language.strip() != "":
                        with st.spinner(f"Translating into {target_language}..."):
                            translated_text = ai_task(user_text, "Translate", target_language)
                        
                        if translated_text:
                            st.subheader(f" Translated Text ({target_language})")
                            st.write(translated_text)
                            combined_output = f"Summary:\n{ai_summary}\n\nTranslation ({target_language}):\n{translated_text}"
                        else:
                            combined_output = f"Summary:\n{ai_summary}"
                    else:
                        combined_output = f"Summary:\n{ai_summary}"
                    
                    st.download_button(
                        label="⬇ Download Outputs",
                        data=combined_output,
                        file_name="text_outputs.txt",
                        mime="text/plain"
                    )

with tab2:
    st.header(" Document Analysis")
    st.write("Upload documents to generate professional reports or emails.")
    
    uploaded_file = st.file_uploader(" Upload a document", type=["txt", "pdf", "docx"], key="doc_upload")
    task_type = st.radio("Select task:", ["Summary Report", "Professional Email", "Translate"], horizontal=True)
    
    # Show language input only when translation is selected
    target_language = None
    if task_type == "Translate":
        target_language = st.text_input(" Enter target language for translation (e.g., German, Chinese):")
    
    if uploaded_file is not None:
        file_text = ""
        
        if uploaded_file.type == "text/plain":
            file_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            file_text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_text = read_docx(uploaded_file)
        
        st.subheader(" Document Content Preview")
        st.text_area("Content", value=file_text[:2000] + ("..." if len(file_text) > 2000 else ""), height=200, disabled=True)
        
        if st.button("Process with AI", key="doc_button"):
            if task_type == "Translate" and not target_language.strip():
                st.warning(" Please enter a target language.")
            else:
                with st.spinner(" AI is analyzing..."):
                    ai_result = ai_task(file_text, task_type, target_language)
                
                if ai_result:
                    st.subheader(f" AI Generated {task_type}")
                    st.write(ai_result)
                    
                    st.download_button(
                        label="⬇ Download Output",
                        data=ai_result,
                        file_name=f"AI_{task_type.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

with tab3:
    st.header(" Dataset Explorer")
    st.write("Upload datasets to get AI-powered analysis with automatic visualizations.")
    
    uploaded_file = st.file_uploader(" Upload a dataset", type=["csv", "xlsx"], key="data_upload")
    
    if uploaded_file is not None:
        df = None
        
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        
        if df is not None:
            st.subheader(" Dataset Preview")
            st.dataframe(df.head())
            
            st.subheader(" Basic Statistics")
            st.write(df.describe())
            
            if st.button(" Analyze with AI"):
                with st.spinner(" AI is analyzing your dataset..."):
                    ai_result = ai_task(df.to_string(), "Dataset Analysis")
                
                if ai_result:
                    st.subheader(" AI Analysis Report")
                    st.write(ai_result)
                    
                    st.download_button(
                        label="⬇ Download Analysis",
                        data=ai_result,
                        file_name="AI_Dataset_Analysis.txt",
                        mime="text/plain"
                    )
            
            # Show visualizations
            show_data_visualizations(df)



        




