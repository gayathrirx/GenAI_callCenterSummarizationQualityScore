import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Call Center Summarization",
    page_icon="🕊️",
)

# Load and display the sidebar HTML
with open("sidebar.html", "r") as file:
    sidebar_html = file.read()

# Streamlit app configuration
# st.set_page_config(page_title='Call Center Transcript Summarization and Quality Score Analysis', layout='wide')
st.title('Call Center Transcript Summarization and Quality Score Analysis')

st.sidebar.success("Select a demo above.")

st.image("images/CallCenterTranscriptDesign.png", caption="Design Flow Diagram")
st.markdown("""
    <style>
    .sidebar-content { 
        # width: 100%; 
        # height: 100%; 
        # overflow: auto; 
    }
    </style>
""", unsafe_allow_html=True)
# Wrap sidebar HTML in a div to manage width and scrolling
components.html(f"<div class='sidebar-content'>{sidebar_html}</div>", height=800, scrolling=True)