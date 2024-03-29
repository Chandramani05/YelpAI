import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Replicate

from utils import set_up_pinnecone, get_template, get_temp_ins
from reviews import upload_embeddings
from restraunts import get_best_restaurants


from torch import cuda, bfloat16
import transformers
import openai
from langchain_community.llms import LlamaCpp
from transformers import TRANSFORMERS_CACHE
from langchain_openai import OpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.agents import load_tools
import re
from langchain.prompts import PromptTemplate, StringPromptTemplate
from typing import List 

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import Union
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import ConversationalRetrievalChain

#print(TRANSFORMERS_CACHE)


import os

REPLICATE_API = os.environ["REPLICATE_API_TOKEN"] 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

rest = get_best_restaurants("Coffee", "DC")

llm = OpenAI(openai_api_key=os.environ['OPEN_AI_KEY'],
             temperature=0.1, max_tokens=1000)
index = set_up_pinnecone()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Pinecone(index, embeddings.embed_query, 'yelp')


tool = create_retriever_tool(
    vectorstore.as_retriever(),
    "Recommendations Search",
    "Searches and returns documents regarding reviews on different businesses in Maryland.",
)
tools = [tool]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
output_parser = CustomOutputParser()





template = get_template()
temp_ins = get_temp_ins()

prompt_Ins = PromptTemplate(
    input_variables=["thought", "query", "observation"],
    template=temp_ins,
)

from langchain_openai import ChatOpenAI

class CustomPromptTemplate(StringPromptTemplate):

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        print(intermediate_steps)
        thoughts = ""
        # Refine the observation
        if len(intermediate_steps) > 0:
            regex = r"Thought\s*\d*\s*:(.*?)\nAction\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*\d*\s*(.*)"
            text_match = intermediate_steps[-1][0].log
            if len(intermediate_steps) > 1:
                text_match = 'Thought: ' + text_match
            print(text_match)
            print("####")
            match = re.search(regex, text_match, re.DOTALL)           
            my_list = list(intermediate_steps[-1])
            p_INS_temp = prompt_Ins.format(thought=match.group(1).strip(), query=match.group(3).strip(), observation=my_list[1])
            my_list[1] = llm(p_INS_temp)
            my_tuple = tuple(my_list)            
            intermediate_steps[-1] = my_tuple
            
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f" {observation}\nThought:"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)
    
template = """
### Instruction: You're a resturant support agent that is talking to a customer. Use only the chat history and the following information
{context}
to answer in a helpful manner to the question. If you don't know the answer - say that you don't know.
Keep your replies short, compassionate and informative.
{chat_history}
### Input: {question}
### Response:
""".strip()
prompt = PromptTemplate(input_variables=["context", "question", "chat_history"],
                              template=template,validate_template=False)
tool_names = [t.name for t in tools]

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
output_parser = CustomOutputParser()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    human_prefix="### Input",
    ai_prefix="### Response",
    output_key="answer",
    return_messages=True,
)

chain = ConversationalRetrievalChain.from_llm(llm, chain_type='stuff', retriever= vectorstore.as_retriever(search_kwargs={'k': 6}), 
                                              memory = memory, combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=True,)

pdf_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

chat_history = []

context = upload_embeddings(restaurant=rest)
print(rest['address'])

question = "What is the address of the restaurant?"
result = pdf_qa.invoke({"question" : question, 'context' : context, "chat_history": chat_history})
print(result['answer'])

# llm_chain = LLMChain(llm=llm, prompt=prompt)

# agent = LLMSingleActionAgent(
#     llm_chain=llm_chain, 
#     output_parser=output_parser,
#     stop=["\nObservation:"], 
#     allowed_tools=tool_names
# )
# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)

# agent_executor.invoke({"input": "what is LangChain?"})
