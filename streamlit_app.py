import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components
from app import run_chain_step, run_chain_step2, process_transcript, summarization_prompt, llm, extract_from_xml_tag, summarization_parser, assessment_prompt, assessment_parser, generate_tables, generate_assessment_tables

# Load and display the sidebar HTML
with open("sidebar.html", "r") as file:
    sidebar_html = file.read()

# Streamlit app configuration
st.set_page_config(page_title='Call Center Transcript Summarization and Quality Score Analysis', layout='wide')
st.title('Call Center Transcript Summarization and Quality Score Analysis')

# Create a two-column layout with equal width
col1, col2 = st.columns(2)  # Two columns of equal width

# Sidebar in the first column
with col1:
    st.image("CallCenterTranscriptDesign.png", caption="Design Flow Diagram")
    st.markdown("""
        <style>
        .sidebar-content { 
            width: 100%; 
            height: 100%; 
            overflow: auto; 
        }
        </style>
    """, unsafe_allow_html=True)
    # Wrap sidebar HTML in a div to manage width and scrolling
    components.html(f"<div class='sidebar-content'>{sidebar_html}</div>", height=600, scrolling=True)

# Main content area in the second column
with col2:
    st.subheader("Upload and Process Data")

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
            st.subheader("Call Assessment and Quality Score")
            st.table(assessment_tables["table_data"])

            # Update status to "Completed"
            status_placeholder.text("Status: Completed")
            progress_bar.progress(100)
        else:
            st.error("Please provide JSON input via file upload or text area.")
