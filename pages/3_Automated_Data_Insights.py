import warnings
import streamlit as st
import matplotlib.pyplot as plt
import csv
import os
import subprocess
from datetime import datetime
import time

from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_openai_functions_agent
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain import hub
from langchain_experimental.tools import PythonREPLTool
from langchain import PromptTemplate


import psycopg2

import os
import psycopg2
from psycopg2 import sql
from sqlalchemy.dialects.postgresql.base import PGDialect

# Suppress all warnings
warnings.filterwarnings("ignore")

PGDialect._get_server_version_info = lambda *args: (9, 2)

model_name = "gpt-3.5-turbo"
# "gpt-4-0125-preview"
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

llm = OpenAI(temperature=0.0)

DATABASE_URL=os.getenv('roach_url2')

sql_db = SQLDatabase.from_uri(DATABASE_URL)

# Define the tables you want to query
tables = [
    "call_assessment_quality_scores",
    "follow_up_actions"
]

def fetch_table_counts():
    """Fetch the count of records from each table."""
    table_counts = {}
    try:
        for table in tables:
            query = f"SELECT count(*) FROM {table};"
            result = sql_db.run(query)
            table_counts[table] = result.strip()  # Assuming the result is a string with the count
    except Exception as e:
        st.error(f"Database connection error: {e}")
    print(table_counts)
    return table_counts



class SQLQueryEngine:
    """
    A class representing an SQL query engine.

    Attributes:
        llm (ChatOpenAI): An instance of ChatOpenAI used for natural language processing.
        toolkit (SQLDatabaseToolkit): An SQL database toolkit instance.
        context (dict): Contextual information obtained from the SQL database toolkit.
        tools (list): List of tools available for SQL query execution.
        prompt (ChatPromptTemplate): The prompt used for interactions with the SQL query engine.
        agent_executor (AgentExecutor): An executor for the SQL query engine's agent.
    """
    def __init__(self, model_name, db):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        self.context = self.toolkit.get_context()
        self.tools = self.toolkit.get_tools()
        self.prompt = None
        self.agent_executor = None

    def set_prompt(self):
        messages = [
            HumanMessagePromptTemplate.from_template("{input} and Use only call_assessment_quality_scores and follow_up_actions tables . Both tables can be joined using columns call_id and csr_id. Column names are duplicated in both tables. Note there are multiple records for the same call_id. For every call attended by the CSR, there will be multiple records that capture the details for that csr_id and call_id. Use group by as needed. Use distinct when needed. Use Join only when needed. Convert score Low, Meidum, High to 3, 2, and 1."),
            AIMessage(content=SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        self.prompt = ChatPromptTemplate.from_messages(messages)
        self.prompt = self.prompt.partial(**self.context)

    def initialize_agent(self):
        agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.toolkit.get_tools(),
            verbose=True,
        )

    def get_query_data(self, query):
        if 'return' in query:
            query = query + "\n" + "Use only call_assessment_quality_scores and follow_up_actions tables . Both tables can be joined using columns call_id and csr_id. Column names are duplicated in both tables. Note there are multiple records for the same call_id. For every call attended by the CSR, there will be multiple records that capture the details for that csr_id and call_id. Use group by as needed. Use distinct when needed. Use Join only when needed. Convert score Low, Medium, High to 3, 2, and 1."
        return self.agent_executor.invoke({"input": query})['output']
    
# REPL -> Read Evaluate Print Loop
class PythonDashboardEngine:
    """
    A class representing a Python dashboard engine.

    Attributes:
        tools (list): A list of tools available for the dashboard engine.
        instructions (str): Instructions guiding the behavior of the dashboard engine.
        prompt (str): The prompt used for interactions with the dashboard engine.
        agent_executor (AgentExecutor): An executor for the dashboard engine's agent.
    """
    def __init__(self):
        self.tools = [PythonREPLTool()]
        self.instructions = """You are an agent designed to write a python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        Always output the python code only.
        """
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        self.prompt = base_prompt.partial(instructions=self.instructions)
        self.agent_executor = None

    def initialize(self):
        agent = create_openai_functions_agent(ChatOpenAI(model=model_name, temperature=0), self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def get_output(self, query):
        print(f"Query: {query}")
        output = self.agent_executor.invoke({"input": "Write a code in python to plot the following data\n\n" + query})
        print(f"Output: {output}")
        return output['output']

    def parse_output(self, inp):
    # Extract the relevant code block
        inp = inp.split('```')[1].replace("```", "").replace("python", "").replace("plt.show()", "")

        # Modify the output to include explicit figure creation and passing it to st.pyplot()
        outp = (
            "import streamlit as st\n"
            "import matplotlib.pyplot as plt\n\n"
            "st.title('LLM Generated Insights')\n"
            "fig, ax = plt.subplots()\n"
            + inp +
            "\nst.pyplot(fig)\n"
        )
        return outp


    def export_to_streamlit(self, data):
        with open("appresult.py", "w") as text_file:
            text_file.write(self.parse_output(data))

        command = "streamlit run appresult.py"
        proc = subprocess.Popen([command], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    def runScript(self, data):
        # safe_globals = {
        #     "__builtins__": None,  # Disable all built-ins for security
        #     "st": st,
        #     "plt": plt,
        # }
        exec(self.parse_output(data))

global sql_query_engine, dashboard_engine

def init_engines():
    sql_query_engine = SQLQueryEngine(model_name, sql_db)
    sql_query_engine.set_prompt()
    sql_query_engine.initialize_agent()
    print("sql query engine initialized")

    dashboard_engine = PythonDashboardEngine()
    dashboard_engine.initialize()
    print("dashboard engine initialized")
    return sql_query_engine, dashboard_engine

# Initialize engines
st.set_page_config(
    page_title="Call Center - Automated Data Insights",
    page_icon="🕊️",
)
st.title("Automated Call Center Data Insights")
with st.expander("Show Design Flow Diagram"):
    st.image("images/dataInsights.png", caption="Design Flow Diagram")
st.write("Use natural language to get insights of the system without writing SQL")

# Initialize the engines once when the Streamlit app starts
if 'sql_query_engine' not in st.session_state:
    sql_query_engine, dashboard_engine = init_engines()
    st.session_state['sql_query_engine'] = sql_query_engine
    st.session_state['dashboard_engine'] = dashboard_engine
else:
    sql_query_engine = st.session_state['sql_query_engine']
    dashboard_engine = st.session_state['dashboard_engine']

# Input box for SQL query
query = st.text_area("Enter your prompt here", value="Number of returns by customer representaive")



# Button to run the query
if st.button("Get Data Insights"):
    progress_bar = st.progress(0)  # Initialize the progress bar at 0%
    try:
        # Update progress bar to 20%
        progress_bar.progress(20)
        # Get query data using SQLQueryEngine
        sql_query_engine_output = sql_query_engine.get_query_data(query)
        st.write("AI generated results:")
        st.write(sql_query_engine_output)

        # Update progress bar to 60%
        progress_bar.progress(60)

        # Generate and display the dashboard
        dashboard_engine_output = dashboard_engine.get_output(sql_query_engine_output)
        
        # Update progress bar to 80%
        progress_bar.progress(80)

        print("dashboard_engine_output", dashboard_engine_output)
        
        dashboard_engine.runScript(dashboard_engine_output)
        
        # Update progress bar to 100%
        progress_bar.progress(100)

        # Complete the progress
        st.success("Data insights generated successfully!")
        
    except Exception as e:
        st.error(f"Error while processing the query: {e}")
        progress_bar.empty()  # Remove the progress bar on error

# if st.button("test"):
#     test_query = "Write a Python code to create a bar plot of [1, 2, 3, 4]."
#     dashboard_engine.get_output(test_query)

# Button to fetch data
if st.button("Show Database Details"):
    progress_bar = st.progress(0)  # Initialize the progress bar at 0%
    try:
        # Update progress bar to 50%
        progress_bar.progress(50)
        table_counts = fetch_table_counts()
        
        # Update progress bar to 100%
        progress_bar.progress(100)

        # Display the counts in a table format
        if table_counts:
            st.write("### Number of Records in Each Table")
            st.table(table_counts)
        else:
            st.write("No data available or there was an error.")
        st.success("Database details fetched successfully!")
        
    except Exception as e:
        st.error(f"Database connection error: {e}")
        progress_bar.empty()  # Remove the progress bar on error