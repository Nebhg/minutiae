import streamlit as st
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import PythonAstREPLTool
import pandas as pd
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

pd.set_option('display.max_rows', 20)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local("advanced_stats", embedding_model)
retriever_tool = create_retriever_tool(vectorstore.as_retriever(), "player_name_search", "Search for a player by name and find the records corresponding to players with similar name as the query")

TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exploratory data analysis to give you more information as needed.

You have a tool called `player_search` through which you can lookup a player by name and find the records corresponding to players with similar name as the query.
You should only really use this if your search term contains a players name. Otherwise, try to solve it with code.

For example:

<question>What is the average points per game for LeBron James?</question>
<logic>Use `player_search` since you can use the query `LeBron James`</logic>

<question>Who has the highest points per game?</question>
<logic>Use `python_repl` since even though the question is about a player, you don't know their name so you can't include it.</logic>
"""

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")

df = pd.read_csv("advanced_stats.csv")
template = TEMPLATE.format(dhead=df.head().to_markdown())

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}")
])

def get_chain():
    repl = PythonAstREPLTool(locals={"df": df}, name="python_repl",
                             description="Runs code and returns the output of the final line",
                             args_schema=PythonInputs)
    tools = [repl, retriever_tool]
    agent = OpenAIFunctionsAgent(llm=ChatOpenAI(temperature=0, model="gpt-4"), prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate")
    return agent_executor

agent_executor = get_chain()

# Streamlit app starts here
st.set_page_config(page_title='NBA API App')
st.title('NBA API App')

query_text = st.text_input('Enter your question:', placeholder = 'Who had the highest points per game?')
# Form input and query
result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Submit')
	if submitted:
		with st.spinner('Calculating...'):
			response = agent_executor({"input": query_text})
			result = response["output"]

if result is not None:
	st.info(result)

st.write(df.head(20))