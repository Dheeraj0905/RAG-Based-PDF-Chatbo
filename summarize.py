import streamlit as st
from transformers import pipeline

# Load the BART model suitable for text summarization
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(long_text):
    # Generate output based on the long text
    output = summarization_model(long_text, max_length=1000, min_length=25, do_sample=False)
    return output[0]['summary_text']

# Streamlit UI
st.title("Text Summarization App")
st.write("Enter the text you want to summarize:")

# Text input
long_text = st.text_area("Long Text", height=300)

# Button to generate summary
if st.button("Summarize"):
    if long_text:
        summary = summarize_text(long_text)
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
