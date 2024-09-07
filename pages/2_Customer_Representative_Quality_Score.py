import streamlit as st
import pandas as pd
import json
from app import run_chain_step, run_chain_step2, process_transcript, summarization_prompt, llm, extract_from_xml_tag, summarization_parser, assessment_prompt, assessment_parser, generate_tables, generate_assessment_tables


# Streamlit app configuration
st.set_page_config(
    page_title="Call Center - Quality Score Analysis",
    page_icon="🕊️",
)

st.header('Quality Score Analysis')

with st.expander("Show Design Flow Diagram"):
    st.image("images/qualityScore.png", caption="Design Flow Diagram")

# Text input for JSON
json_text = st.text_area("Or paste JSON text here:", height=300)

# Status label
status_placeholder = st.empty()
progress_bar = st.empty()

# Button to submit and process the data
if st.button('Submit'):
    if json_text:
        # Update status to "Processing"
        status_placeholder.text("Status: Processing")
        progress_bar.progress(0)  # Initialize progress bar

        # Process transcript and run summarization chain
        processed_transcript = process_transcript(json_text)
        progress_bar.progress(20)

        # Run assessment chain
        response2 = run_chain_step2(
            processed_transcript,
            assessment_prompt,
            llm,
            extract_from_xml_tag,
            assessment_parser
        )
        progress_bar.progress(90)

        assessment_tables = generate_assessment_tables(response2)
        
        # Display assessment table
        st.subheader("Quality Score")
        st.table(assessment_tables["table_data"])

        # Update status to "Completed"
        status_placeholder.text("Status: Completed")
        progress_bar.progress(100)
    else:
        st.error("Please provide JSON input via file upload or text area.")
