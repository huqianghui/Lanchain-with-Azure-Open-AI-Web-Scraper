from openAIRoundRobin import get_openaiByRoundRobinMode
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQAWithSourcesChain
from webConentRetriever import WebContent2LocalFileSplitRetriever
from dotenv import load_dotenv
import os
import json
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# 加载 .env 文件中的环境变量
load_dotenv(dotenv_path)

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.

The original question is in Chinese, but the context is in English.
You can translate the question to English if you want to.
The original question is the list of attributes which want to be queried.like this: ["testA","testB"]
then your output format will be like this:

[{{
  "index": "testA",  
  "value": "FLEXISPOT"
}},
{{
  "index": "testB",  
  "value": "4.7"
}}
]

In the json result, you should answer all the attributes in the question list.If you don't know the answer, just set the value to ' I don't konw.', don't try to make up an answer.

SOURCES:

QUESTION: {question}
=========
{summaries}
=========
ANSWER:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
qaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="stuff", 
                                         retriever=WebContent2LocalFileSplitRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

# result = qaChain('''["品牌"]''')
# print("AzureCognitiveSearchMockRetriever: ")
# print(result)

def get_content_by_index_contents(index_contents:[str]) -> str:
    try:
        result = qaChain(index_contents)
        print(result["answer"])
        return result["answer"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
