from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os

load_dotenv()

def get_llm():
    return init_chat_model(model= "groq:openai/gpt-oss-20b")


def split_transcript(transcript : str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summerizes(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "summarizes this portion of a meeting transcript concisely."),
            ("human", "{text}")
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summerizes = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summerizes)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            "You are an expert meeting summerizer. Combine these partial simmaries "
            "into one final professional meeting summery in bullet point"
            ),

            ("human", "{text}")
        ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript : str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text":x}) | 
        ChatPromptTemplate.from_messages([
            ("system",
            "Based on the meeting transcript, generate a short professional meeting title"
            "(max 8 word). only return the title, nothing else."
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])