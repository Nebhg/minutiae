import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

df = pd.read_csv('titanic.csv')


llm = ChatOpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df, agent_type=AgentType.OPENAI_FUNCTIONS)


import streamlit as st
# Streamlit app starts here
st.set_page_config(page_title='🦜🔗 Ask the NBA API App')
st.title('🦜🔗 Ask the NBA API App')

query_text = st.text_input('Enter your question:', placeholder = 'Who had the highest AST_TOV ratio across all game?')
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