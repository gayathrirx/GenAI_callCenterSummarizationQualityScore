from enum import Enum
import json
import re
from pathlib import Path
from functools import partial
from typing import List, Dict

from pydantic import BaseModel, Field

import anthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain.output_parsers import PydanticOutputParser
from rich import print as rprint
from rich.markdown import Markdown

from tabulate import tabulate
import pandas as pd


import os

# Utility function to pretty-print JSON
def pretty_print_json(data):
    if isinstance(data, dict):
        data = json.dumps(data, indent=2)
    elif isinstance(data, str):
        try:
            json.loads(data)  # Validate JSON format
            data = json.dumps(json.loads(data), indent=2)
        except json.JSONDecodeError:
            pass  # Not JSON
    # rprint(Markdown(f"```json\n{data}\n```"))

# Read call transcript file
transcripts = Path("data")
call_transcript_file = "sample1.json"
transcript_path = transcripts / call_transcript_file

with open(transcript_path, "r") as f:
    transcript = f.read()

transcript_dict = json.loads(transcript)
call_date = transcript_dict['call_ID']
call_time = transcript_dict['CSR_ID']
call_transcript = transcript_dict['call_transcript']
call_date = transcript_dict['call_date']
call_time = transcript_dict['call_time']

# pretty_print_json(transcript_dict)
# print(call_transcript)

def extract_from_xml_tag(response: str, tag: str) -> str:
    tag_txt = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
    if tag_txt:
        return tag_txt.group(1)
    else:
        # print(response)
        return ""

class CallSummary(BaseModel):
    call_summary: str = Field(description="Call transcript summary: ")
    key_takeaways: List[str] = Field(description="Call transcript key takeaways: ")
    follow_up_actions: List[str] = Field(description="Call Transcript key action items: ")

summarization_parser = PydanticOutputParser(pydantic_object=CallSummary)

summarization_template = """
Please provide a summary of the following call transcript provided between <transcript></transcript> tags. 

Capture key takeaways and specific follow up actions. 

Follow up actions can be categorized as 'Initiate refund' ,'Follow up with customer', 'Process improvement' and 'Others', 

Skip the preamble and go straight to the answer. <transcript>{call_transcript}</transcript> 

Format your response per the instructions below: {format_instructions} 

Place your response between <output></output> tags. 

\n\nAssistant:

"""


summarization_prompt = ChatPromptTemplate.from_template(
    summarization_template,
    partial_variables={
        "format_instructions": summarization_parser.get_format_instructions()
    },
)

# rprint(summarization_prompt.dict())

def process_transcript(transcript: str) -> str:
    json_transcript = json.loads(transcript)
    call_transcript = "\n".join(json_transcript.get("call_transcript", []))
    return call_transcript

# Initialize the ChatAnthropic client
class ChatAnthropic:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def __call__(self, messages):
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,  # Pass messages as a list of dictionaries
            max_tokens=8000,  # Correct parameter name for max tokens
            temperature=0.4,
            top_p=1
        )
        # print("!!!!!!!!!!",response)
        # print("$$$$$$$$$$",response.content[0].text)
        match = re.search(r'<output>\n({.*?})\n</output>', response.content[0].text, re.DOTALL)
        if match:
            json_string = match.group(1)
        # print("@@@@@@@@@",json_string)
        return json_string

llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", 
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                    )

def create_messages(prompt: str) -> List[Dict[str, str]]:
    # Create messages list
    return [{"role": "user", "content": prompt}]

# Define the components for the chain
def run_chain_step(process_transcript, summarization_prompt, llm, extract_from_xml_tag, parser):
    processed_transcript = process_transcript(transcript)
    formatted_prompt = summarization_prompt.format(call_transcript=processed_transcript)
    messages = create_messages(formatted_prompt)
    response = llm(messages)
    # extracted_output = extract_from_xml_tag(response, tag="output")
    # parsed_output = parser.parse_raw(extracted_output)
    return response

# Run the chain
# pretty_print_json(transcript)
# print("Running summarization chain...")

# response = run_chain_step(
#     process_transcript,
#     summarization_prompt,
#     llm,
#     extract_from_xml_tag,
#     summarization_parser
# )
# data = json.loads(response)
# pretty_print_json(summary)

# Save call summary to database - Summary - TODO
# Prepare data for tables
# summary_data = [
#     ["Call Summary", data["call_summary"]]
# ]

# key_takeaways_data = [
#     [f"{i + 1}.", takeaway]
#     for i, takeaway in enumerate(data["key_takeaways"])
# ]

# Separate follow-up actions by category
category_actions = {
    "Initiate refund": [],
    "Process improvement": [],
    "Follow up with customer": [],
    "Others": []
}

# for action in data["follow_up_actions"]:
#     for category in category_actions:
#         if action.startswith(category):
#             category_actions[category].append(action)
#             break

# Convert to table data format
# def convert_to_table_data(category, actions):
#     return [[category, action] for action in actions]

# # Create table data for each category
# follow_up_tables = {
#     category: convert_to_table_data(category, actions)
#     for category, actions in category_actions.items() if actions
# }

# Define column headers
# summary_headers = ["Description", "Details"]
# key_takeaways_headers = ["#", "Key Takeaway"]
# follow_up_headers = ["Category", "Action"]

# # Print tables
# print("Call Summary:")
# print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))

# print("\nKey Takeaways:")
# print(tabulate(key_takeaways_data, headers=key_takeaways_headers, tablefmt="grid"))

# print("\nFollow-Up Actions:")
# for category, table_data in follow_up_tables.items():
#     print(f"\n{category} Actions:")
#     print(tabulate(table_data, headers=follow_up_headers, tablefmt="grid"))

assessment_template = """
Evaluate call transcript against categories shown between <categories></categories> tags and provide score as 'High', 'Medium', 'Low' for each category.
Skip the preamble and go straight to the answer.

<categories>
1. Communication Skills:
 - Clarity: How clearly and concisely does the CSR communicate information?
 - Active Listening: Does the CSR actively listen to the customer's concerns and questions?
 - Empathy: How well does the CSR demonstrate empathy and understanding towards the customer?

2. Problem Resolution:
 - Effectiveness: How well did the CSR resolve the customer's issue or answer their question?
 - Timeliness: Was the issue resolved in a reasonable amount of time?

3. Product Knowledge:
 - Familiarity: Does the CSR have a good understanding of the company's products and services?
 - Accuracy: How accurate and precise are the answers provided by the CSR?

4. Professionalism:
 - Tone and Manner: How professional is the tone and manner of the CSR throughout the call?
 - Courtesy: Does the CSR maintain a courteous and respectful attitude towards the customer?

5. Problem Escalation:
 - Recognition: Did the CSR recognize when an issue required escalation to a higher level of support?
 - Handoff: How smoothly and effectively did the CSR transfer the call if escalation was necessary?
<categories>

Here is the call transcript:
<transcript>{transcript}</transcript>

Format your response per the instructions below: 
{format_instructions} 

Place your response between <output></output> tags. 

"""
# Define a data schema for the LLM output with required attributes

class ScoreValue(Enum):
    High = "High"
    Medium = "Medium"
    Low = "Low"

class Score(BaseModel):
    score: ScoreValue
    score_explanation: str

class Evaluation(BaseModel):
    Communication_Skills: Score
    Problem_Resolution: Score
    Product_Knowledge: Score
    Professionalism: Score
    Problem_Escalation: Score
    
# Define Pydantic parser based on data schema 
assessment_parser = PydanticOutputParser(pydantic_object=Evaluation)
# Incorporate the format instructions into the LLM prompt based on the prompt template
assessment_prompt = ChatPromptTemplate.from_template(
    assessment_template,
    partial_variables={
        "format_instructions": assessment_parser.get_format_instructions()
    },
)

def run_chain_step2(process_transcript, assessment_prompt, llm, extract_from_xml_tag, parser):
    processed_transcript = process_transcript(transcript)
    formatted_prompt = assessment_prompt.format(transcript=processed_transcript)
    messages = create_messages(formatted_prompt)
    response = llm(messages)
    # extracted_output = extract_from_xml_tag(response, tag="output")
    # parsed_output = parser.parse_raw(extracted_output)
    return response

# Construct the chain by essembing all required components via LangChain

# response2 = run_chain_step2(
#     process_transcript,
#     assessment_prompt,
#     llm,
#     extract_from_xml_tag,
#     assessment_parser
# )
# # Preview content of the call_assessment JSON object
# call_assessment = json.loads(response2)
# # pretty_print_json(call_assessment)

# # Preview score values for each category provided in the LLM output
# # for category, details in call_assessment.items():
# #         score = details.get("score", "N/A")
# #         explanation = details.get("score_explanation", "No explanation provided")
# #         print(f"{category}: score={score}, explanation={explanation}")
# # Prepare data for tabulate
# table_data = [
#     [category, details['score'], details['score_explanation']]
#     for category, details in call_assessment.items()
# ]

# # Define column headers
# headers = ["Category", "Score", "Explanation"]

# # Print table
# print("Call Assessment and Quality Score")
# print(tabulate(table_data, headers=headers, tablefmt="grid"))

def generate_tables(response):
    data = json.loads(response)
    
    summary_data = pd.DataFrame([["Call Summary", data["call_summary"]]], columns=["Description", "Details"])
    key_takeaways_data = pd.DataFrame(
        [[f"{i + 1}.", takeaway] for i, takeaway in enumerate(data["key_takeaways"])],
        columns=["#", "Key Takeaway"]
    )
    
    # Separate follow-up actions by category
    category_actions = {
        "Initiate refund": [],
        "Process improvement": [],
        "Follow up with customer": [],
        "Others": []
    }
    
    for action in data["follow_up_actions"]:
        for category in category_actions:
            if action.startswith(category):
                category_actions[category].append(action)
                break

    follow_up_tables = {
        category: pd.DataFrame([[category, action] for action in actions], columns=["Category", "Action"])
        for category, actions in category_actions.items() if actions
    }
    
    return {
        "summary_data": summary_data,
        "key_takeaways_data": key_takeaways_data,
        "follow_up_tables": follow_up_tables
    }

def generate_assessment_tables(response2):
    call_assessment = json.loads(response2)
    table_data = pd.DataFrame([
        [category, details['score'], details['score_explanation']]
        for category, details in call_assessment.items()
    ], columns=["Category", "Score", "Explanation"])

    return {
        "table_data": table_data
    }
