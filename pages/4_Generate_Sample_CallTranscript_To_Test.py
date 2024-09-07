from openai import OpenAI
import streamlit as st
import random
import json
from datetime import datetime, timedelta
import os
import gtts
from gtts import gTTS

from persist_data import insert_call_data
# from playsound import playsound

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

# Streamlit app title

st.set_page_config(
    page_title="Call Center - Test Data Generator",
    page_icon="🕊️",
)

st.title("Dynamic Customer Service Call Generator")


# Instructions
st.markdown("""
## How to use:
1. **Enter your prompt**: Customize it based on your scenario.
2. **Choose the GPT Model**: Select a model for generating the response.
3. **Adjust the max tokens and temperature**: Control the response length and creativity.
4. **Generate Transcript**: Generate and view the call transcript.
5. **Copy the Output**: Simply select and copy the JSON output from the textarea below.
""")

# Default prompt text, dynamically constructing CSR_IDs into the prompt
default_prompt = """
Generate a realistic customer service call transcript between a customer service representative (CSR) and a customer. 
The CSR should have a full name in the format FirstNameLastNameID, where ID is randomly chosen from a list of given IDs. 
The CSR is helping a customer who has called to report an issue with their internet or cable service. 
The issue should be generated dynamically by the model. 
The CSR should handle the call professionally, ask for necessary details like the customer's name, account number, and offer a solution or compensation such as a bill credit. 
The conversation should include pleasantries, an apology for the issue, and an appropriate resolution.
"""

# Model options for dropdown
model_options = ["gpt-3.5-turbo-instruct", "text-davinci-003", "text-curie-001", "gpt-3.5-turbo", "gpt-4"]
selected_model = st.selectbox("Choose the GPT Model:", model_options)

# Input fields for max_tokens and temperature
max_tokens = st.number_input("Max Tokens:", min_value=50, max_value=4000, value=600)
temperature = st.slider("Temperature (Creativity):", min_value=0.0, max_value=1.0, value=0.7)

# Text area for custom prompt
prompt = st.text_area("Enter your prompt here:", value=default_prompt, height=250)

# Button to generate transcript
if st.button("Generate Call Transcript"):

    # Function to convert text to speech and save as an audio file
    def text_to_speech(text, output_file):
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        return output_file

    # Function to generate and handle the call scenario
    def handle_call_scenario():
        call_scenario = generate_call_scenario(selected_model, prompt, max_tokens, temperature)  # Assuming this function is already defined and works
        json_output = json.dumps(call_scenario, indent=4)
        st.text_area("JSON Output:", value=json_output, height=300, help="Select and copy the JSON from here.")
        
        # Combining all dialogue into a single string to convert to speech
        full_text = " ".join(call_scenario["call_transcript"])
        audio_file = text_to_speech(full_text, 'call_scenario.mp3')
        
        # Optionally play the audio file directly in the app
        # playsound(audio_file)
        
        # Display a link to download the audio file
        st.audio(audio_file)
        insert_call_data(call_scenario, open(audio_file, 'rb').read())
        st.write("Call transcript saved to database.")
        
    def generate_audio_from_text(call_scenario):
        try:
            # Define the voices for CSR and Customer
            csr_voice = "alloy"  # Example voice for CSR
            customer_voice = "sunday"  # Example voice for Customer

            # Prepare the conversation with voice specifications
            ssml_parts = []
            for line in call_scenario['call_transcript']:
                if line.startswith("CSR:"):
                    # Strip the "CSR:" part and wrap the line with the voice tag for the CSR
                    ssml_parts.append(f"<voice name='{csr_voice}'>{line[4:].strip()}</voice>")
                else:
                    # Strip the "Customer:" part and wrap the line with the voice tag for the Customer
                    ssml_parts.append(f"<voice name='{customer_voice}'>{line[9:].strip()}</voice>")

            # Join all parts into a single SSML string
            ssml_text = "<speak>" + " ".join(ssml_parts) + "</speak>"

            # Using OpenAI's text-to-speech model to generate audio
            response = client.audio.speech.create(
                # client.Audio.create(
                model="tts-1",  # Replace 'tts-1' with the specific model you intend to use
                input=ssml_text
                # voice="alloy"
                # use_ssml=True  # This enables SSML processing
            )
            audio_url = response['url']  # The URL to the generated audio file
            return audio_url
        except Exception as e:
            return str(e)
    
    # Function to generate a random scenario with OpenAI GPT
    def generate_call_scenario(selected_model, prompt, max_tokens, temperature):
        # Randomly select a CSR_ID from the predefined list
        csr_ids = ['JohnDoe101', 'JaneDoe102', 'AliceSmith103', 'BobBrown104', 'CharlieBlack105', 'DianaWhite106', 'EvanGreen107', 'FionaGrey108', 'GeorgeHill109', 'HannahStorm110']
        csr_id = random.choice(csr_ids)
        
        # Modify prompt to include selected CSR_ID
        prompt_with_id = f"{prompt}\nThe CSR_ID for this call is {csr_id}."

        # Generate random call ID and time
        call_id = str(random.randint(10000, 99999))
        call_time = datetime.now() - timedelta(minutes=random.randint(0, 60))
        call_date = call_time.strftime("%Y-%m-%d")
        call_time_str = call_time.strftime("%H:%M:%S")
        
        # Call the OpenAI API to generate the transcript using the selected model
        response = client.completions.create(
            model=selected_model,
            prompt=prompt_with_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=None
        )
        
        # Extract the response and format it
        transcript = response.choices[0].text.strip().split("\n")

        # Create the output JSON
        call_scenario = {
            "call_transcript": [line.strip() for line in transcript if line.strip()],
            "call_ID": call_id,
            "CSR_ID": csr_id,  # Ensure the selected CSR_ID is used
            "call_date": call_date,
            "call_time": call_time_str
        }

        return call_scenario
        
    # Call the function to generate the call scenario
    handle_call_scenario()

    # # Generate and display the call scenario
    # call_scenario = generate_call_scenario(selected_model, prompt, max_tokens, temperature)
    # json_output = json.dumps(call_scenario, indent=4)
    # st.text_area("JSON Output:", value=json_output, height=300, help="Select and copy the JSON from here.")

    # # Generate audio from the conversation transcript
    # audio_url = generate_audio_from_text(call_scenario)

    # if audio_url.startswith('http'):
    #     st.audio(audio_url)
    # else:
    #     st.error(f"Failed to generate audio: {audio_url}")
