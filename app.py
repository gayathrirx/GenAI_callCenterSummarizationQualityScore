from enum import Enum
import json
import re
from pathlib import Path
from functools import partial
from typing import List, Dict

from persist_data import insert_assessment_data, insert_follow_up_actions
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
# transcripts = Path("data")
# call_transcript_file = "sample1.json"
# transcript_path = transcripts / call_transcript_file

# with open(transcript_path, "r") as f:
#     transcript = f.read()

# transcript_dict = json.loads(transcript)
call_ID = "unknown"
CSR_ID = "unknown"
# call_transcript = transcript_dict['call_transcript']
# call_date = transcript_dict['call_date']
# call_time = transcript_dict['call_time']

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
Provide a summary of the following call transcript provided between <transcript></transcript> tags. 

Generate key takeaways and specific follow up actions. 

Categorize the follow up actions as 'Initiate refund' ,'Follow up with customer', 'Process improvement' and 'Others', 

Skip the preamble and go straight to the answer. 
Include details of refund and timing if discussed in the call for 'Initiate refund' category.
Include timing for follow up  if discussed in the call for 'Follow up with customer' category.
Include just one Process improvement action for 'Process improvement' category.

<transcript>{call_transcript}</transcript> 

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
    global call_ID, CSR_ID
    json_transcript = json.loads(transcript)
    call_transcript = "\n".join(json_transcript.get("call_transcript", []))
    call_ID = json_transcript.get('call_ID')
    CSR_ID = json_transcript.get('CSR_ID')
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
def run_chain_step(processed_transcript, summarization_prompt, llm, extract_from_xml_tag, parser):
    # processed_transcript = process_transcript(transcript)
    formatted_prompt = summarization_prompt.format(call_transcript=processed_transcript)
    messages = create_messages(formatted_prompt)
    response = llm(messages)
    # extracted_output = extract_from_xml_tag(response, tag="output")
    # parsed_output = parser.parse_raw(extracted_output)
    return response

# Separate follow-up actions by category
category_actions = {
    "Initiate refund": [],
    "Process improvement": [],
    "Follow up with customer": [],
    "Others": []
}

assessment_template = """
Evaluate call transcript against categories shown between <categories></categories> tags and provide score as 'High', 'Medium', 'Low' for each category.
Skip the preamble and go straight to the answer in one sentence per category.

<categories>
1. Communication Skills:
   - Message Delivery: How clearly and effectively does the representative convey information to the customer?
   - Engagement: To what extent does the representative actively engage with the customer's needs and concerns?

2. Issue Resolution:
   - Problem-Solving: How effectively does the representative address and resolve the customer's issue?
   - Efficiency: Was the issue handled in a timely and efficient manner?

3. Product Expertise:
   - Knowledge Depth: Does the representative exhibit a deep understanding of the company's offerings?
   - Response Accuracy: How precise and correct are the representative's responses regarding the products or services?

4. Professional Conduct:
   - Tone and Delivery: How professional and appropriate is the representative's tone and delivery throughout the call?
   - Respectfulness: Does the representative consistently demonstrate respect and courtesy towards the customer?

5. Escalation Handling:
   - Escalation Appropriateness: Did the representative correctly identify when to escalate an issue?
   - Smooth Transition: How effectively does the representative manage the escalation process or transition to another department?

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

def run_chain_step2(processed_transcript, assessment_prompt, llm, extract_from_xml_tag, parser):
    # processed_transcript = process_transcript(transcript)
    formatted_prompt = assessment_prompt.format(transcript=processed_transcript)
    messages = create_messages(formatted_prompt)
    response = llm(messages)
    # extracted_output = extract_from_xml_tag(response, tag="output")
    # parsed_output = parser.parse_raw(extracted_output)
    return response

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

    # follow_up_tables = {
    #     category: pd.DataFrame([[category, action] for action in actions], columns=["Category", "Action"])
    #     for category, actions in category_actions.items() if actions
    # }

    follow_up_tables = {
    category: pd.DataFrame(
        [[call_ID, CSR_ID, category, action] for action in actions],
        columns=["Call ID", "CSR ID", "Category", "Action"]
    )
    for category, actions in category_actions.items() if actions
    }

    #Persist data into follow_up_actions table
    insert_follow_up_actions(follow_up_tables)
    
    return {
        "summary_data": summary_data,
        "key_takeaways_data": key_takeaways_data,
        "follow_up_tables": follow_up_tables
    }

def generate_assessment_tables(response2):
    call_assessment = json.loads(response2)
    table_data = pd.DataFrame([
        [call_ID, CSR_ID, category, details['score'], details['score_explanation']]
        for category, details in call_assessment.items()
    ], columns=["Call ID", "CSR ID", "Category", "Score", "Explanation"])

    insert_assessment_data(table_data)

    return {
        "table_data": table_data
    }
