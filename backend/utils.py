
import os
from pinecone import Pinecone, ServerlessSpec,PodSpec


def set_up_pinnecone() :
    pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index('yelp')
    return index



def get_template() :
    template =  """<s>[INST] <<SYS>>
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    <</SYS>>
    Answer the following questions as best you can. You have access to the following tools:
    Recommendations Search: A database containing information on all stores in the given location. Useful for when you need to find recommendations for a user. The input is the question to search relevant information.

    Strictly use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Google Search]
    Action Input: the input to the action, should be a question.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Here is the user query:
    {input} [/INST]
    {agent_scratchpad}"""
    return template


def get_temp_ins() :
    temp_ins = """<s>[INST] <<SYS>>
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    <</SYS>>
    Question: {thought}
    Query: {query}
    Observation: {observation}

    Here is the user query:
    Make a short summary of useful information from the result observation that is related to the question. Please don't provide any other information and don't make up stuff.
     Also don't recommend anything [/INST]"""
    return temp_ins
