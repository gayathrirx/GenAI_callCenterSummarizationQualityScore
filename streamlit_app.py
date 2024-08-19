import streamlit as st
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

    # File upload
    uploaded_file = st.file_uploader("Upload a JSON file", type="json")

    #default text
    default_text = """
    {
    "call_ID": "12345",
    "CSR_ID": "JaneDoe123",
    "call_date": "2024-02-01",
    "call_time": "02:16:43",
    "call_transcript": [
        "CSR: Thank you for calling ABC Travel, this is Jane. How may I assist you today? ",
        "Customer: Yes, I need help with a reservation I made last week. This is unacceptable service! ",
        "CSR: I apologize for the trouble. May I have your name and reservation number to look up your booking? ",
        "Customer: It's John Smith. My reservation number is 012345. I booked a trip to Hawaii last week and just got an email that my flight was canceled! This is ridiculous. ",
        "CSR: Let me take a look at your reservation here Mr. Smith. I see that your flight from Chicago to Honolulu on March 15th was indeed canceled by the airline. I do apologize for this inconvenience. ",
        "Customer: This is unbelievable! I booked this trip months ago. How could you just cancel my flight like that? I took time off work and made so many plans. This is completely unacceptable! ",
        "CSR: You're absolutely right, having a flight canceled can be very disruptive. As your travel agent, I want to do everything I can to get this fixed for you right away. It looks like the airline has rebooked you on a flight that leaves a few hours later on the same day. I know that's still an inconvenience though. Let me see what other options may be available. ",
        "Customer: This is ridiculous. I should get a full refund if you're going to cancel my flight like that. I don't want another flight, I just want my money back! ",
        "CSR: I completely understand your frustration, Mr. Smith. Since this cancellation was initiated by the airline, you are entitled to a full refund if you prefer not to be rebooked. I can definitely process that refund for the flight cost right away. How about the hotel and other portions of your trip - would you like for me to look into refunds or changes for those as well? My goal is to make sure you are completely satisfied. ",
        "Customer: This is unacceptable. I spent so much money on this trip and now it's ruined. I want a full refund for everything - the flight, the hotel, the car rental. You need to fix this! ",
        "CSR: You're absolutely right, Mr. Smith. Let me process full refunds for your entire trip booking right now. I see you booked 2 roundtrip flights, 5 nights hotel in Honolulu, and a 7 day car rental. I will get all of those refunded in full immediately. You should see the refund hit your credit card in 3-5 business days. I sincerely apologize that we had to cancel a portion of your trip. Providing a seamless travel experience is our top priority, so I appreciate you bringing this issue to my attention. ",
        "Customer: How could you let this happen? I booked my trip so far in advance specifically to avoid problems! Now everything is ruined and I had to waste my time calling you to get this fixed. This is the worst service ever. ",
        "CSR: Mr. Smith, I fully understand why you are upset about having your trip canceled. As a valued customer, you should be able to trust that your travel plans will go smoothly when you book with us. This situation absolutely falls short of our service standards. To make things right, I would like to offer you a $200 travel voucher that can be used on a future trip as an apology for this major inconvenience. Would that help restore your confidence in our company? ",
        "Customer: I don't want a voucher, I just want you to do your job! This is unbelievable. I need to speak to a supervisor immediately. ",
        "CSR: I certainly understand you wishing to speak to a supervisor to express your frustrations about this situation. Please hold for just one moment while I transfer you. Again, I sincerely apologize that we failed to meet expectations on this booking. We value you as our customer and want to regain your trust. Please hold and a supervisor will be right with you. ",
        "Supervisor: Hello Mr. Smith, this is Sarah the supervisor. I understand you've had trouble with your recent booking to Hawaii. I want to sincerely apologize for the cancellation - I know how disruptive that must be. Jane briefed me on the situation and I see she processed full refunds for your trip. I completely understand your frustration. At ABC Travel, it is our top priority to deliver seamless travel experiences to our valued customers like yourself. We clearly dropped the ball and I take full responsibility for that. What else can I do to help restore your confidence in us moving forward? I'm happy to apply a credit for a future trip or look into any other options. ",
        "Customer: This has been a terrible experience. You should train your staff better so these problems don't happen. I expect much better service than this if I'm going to book through your company again. ",
        "Supervisor: You're absolutely right, Mr. Smith. The cancellation of your trip should not have happened. This is clearly an area where we need to improve our service and internal procedures. I will work with our team to assess what went wrong and implement better training around managing cancellations and rebookings. We value you as our customer and want to learn from this experience. I sincerely appreciate you taking the time to speak with me directly so we can improve. Please feel free to reach out to me personally anytime if you do choose to book future travel with ABC. My goal is to restore your confidence in us."
    ]
    }
    """

    # Text input for JSON
    json_text = st.text_area("Or paste JSON text here:", value=default_text, height=300)

    # Button to submit and process the data
    if st.button('Submit'):
        if uploaded_file is not None:
            json_text = uploaded_file.read().decode("utf-8")

        if json_text:
            st.write("Processing...")

            # Run summarization chain
            response = run_chain_step(
                process_transcript,
                summarization_prompt,
                llm,
                extract_from_xml_tag,
                summarization_parser
            )
            
            summary_tables = generate_tables(response)
            
            # Display tables
            st.subheader("Call Summary:")
            st.table(summary_tables["summary_data"])

            st.subheader("Key Takeaways:")
            st.table(summary_tables["key_takeaways_data"])

            st.subheader("Follow-Up Actions:")
            for category, table_data in summary_tables["follow_up_tables"].items():
                st.write(f"{category} Actions:")
                st.table(table_data)

            # Run assessment chain
            response2 = run_chain_step2(
                process_transcript,
                assessment_prompt,
                llm,
                extract_from_xml_tag,
                assessment_parser
            )
            
            assessment_tables = generate_assessment_tables(response2)
            
            # Display assessment table
            st.subheader("Call Assessment and Quality Score")
            st.table(assessment_tables["table_data"])
        else:
            st.error("Please provide JSON input via file upload or text area.")
