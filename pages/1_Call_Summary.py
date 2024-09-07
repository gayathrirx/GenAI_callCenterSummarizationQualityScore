import streamlit as st
import pandas as pd
import json
from app import run_chain_step, run_chain_step2, process_transcript, summarization_prompt, llm, extract_from_xml_tag, summarization_parser, assessment_prompt, assessment_parser, generate_tables, generate_assessment_tables


# Streamlit app configuration
st.set_page_config(
    page_title="Call Center - Call Summarization",
    page_icon="🕊️",
)

st.header('Call Center Transcript Summarization')


with st.expander("Show Design Flow Diagram"):
    st.image("images/callSummary.png", caption="Design Flow Diagram")

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

        response = run_chain_step(
            processed_transcript,
            summarization_prompt,
            llm,
            extract_from_xml_tag,
            summarization_parser
        )
        progress_bar.progress(50)

        summary_tables = generate_tables(response)
        progress_bar.progress(70)
        
        # Display tables
        st.subheader("Call Summary:")
        st.table(summary_tables["summary_data"])

        st.subheader("Key Takeaways:")
        st.table(summary_tables["key_takeaways_data"])

        combined_table = []

        for category, table_data in summary_tables["follow_up_tables"].items():
            # Add a column to identify the category
            table_data["Category"] = category
            # Append the dataframe to the list
            combined_table.append(table_data)

        # Concatenate all the tables into a single dataframe
        combined_df = pd.concat(combined_table)

        # Display the combined table
        st.write("Follow-Up Actions:")
        st.table(combined_df)

        # Update status to "Completed"
        status_placeholder.text("Status: Completed")
        progress_bar.progress(100)
    else:
        st.error("Please provide JSON input via file upload or text area.")
