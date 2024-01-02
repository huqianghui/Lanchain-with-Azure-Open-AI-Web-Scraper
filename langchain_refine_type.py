from openAIRoundRobin import get_openaiByRoundRobinMode
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQAWithSourcesChain
from webConentRetriever import WebContent2LocalFileSplitRetriever
from dotenv import load_dotenv
import os
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# 加载 .env 文件中的环境变量
load_dotenv(dotenv_path)

from langchain.prompts import PromptTemplate

keywords_prompt_template = """Use the following pieces of context to answer the question at the end. 
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.

The original question is in Chinese, but the context is in English.
You can translate the question to English if you want to.
The original question is the list of attributes which want to be queried.like this: ["testA","testB"]

Do not include special characters, such as ",' and so on in the values of the answer.
That may cause the json format to be invalid. Your answer should be in json format. Don't include other words in your answer.
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
The context below.\n
------------\n
{context_str}\n
------------\n

After a comprehensive analysis of the entire page, the main focus pertains to user search terms related to products (search terms used when people intend to make a purchase).

answer the question:: {question} \n
"""

KEYWORD_QUESTION_PROMPT = PromptTemplate(
    template=keywords_prompt_template, input_variables=["question", "context_str"]
)


keyword_refine_prompt_template = """Use the following pieces of context to answer the question at the end. 
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.

The original question is the list of attributes which want to be queried.like this: ["testA","testB"]

Do not include special characters, such as ",' and so on in the values of the answer.
That may cause the json format to be invalid. Your answer should be in json format. Don't include other words in your answer.
then your output format must be valid json format,it will be like this:

[{{
  "index": "testA",  
  "value": "FLEXISPOT"
}},
{{
  "index": "testB",  
  "value": "4.7"
}}
]

The original question is as follows: {question} \n

We have provided an existing answer, including sources: {existing_answer}\n
We have the opportunity to refine the existing answer
(only if needed) with some more context below.\n
------------\n
{context_str}\n
------------\n

The original question is in Chinese, but the context is in English.
You can translate the question to English if you want to.

After a comprehensive analysis of the entire page, the main focus pertains to user search terms related to products (search terms used when people intend to make a purchase).

Given the new context, refine the original answer to better 
answer the question. 
If you do update it, please update the sources as well.
If the context isn't useful, return the original answer.
"""

KEYWORD_REFINE_PROMPT = PromptTemplate(
    template=keyword_refine_prompt_template, input_variables=["question", "existing_answer", "context_str"]
)
chain_type_kwargs = {"question_prompt":KEYWORD_QUESTION_PROMPT,"refine_prompt": KEYWORD_REFINE_PROMPT}
keywordsQaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="refine", 
                                         retriever=WebContent2LocalFileSplitRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

# result = qaChain("产品标题")
# print("WebContentSplitRetriever: ")
# print(result)
# step 3) insert result into excel column, if result is valid number

def get_keywords(index_contents:[str]) -> str:
    try:
        result = keywordsQaChain(index_contents)
        print(result["answer"])
        return result["answer"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


pros_cons_prompt_template = """Use the following pieces of context to answer the question at the end. 
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.

The original question is in Chinese, but the context is in English.
You can translate the question to English if you want to.
The original question is the list of attributes which want to be queried.like this: ["testA","testB"]

Do not include special characters, such as ",' and so on in the values of the answer.
That may cause the json format to be invalid.Your answer should be in json format. Don't include other words in your answer.
The words of your answer should be in English.
then your output format must be valid json format,it will be like this:

[{{
  "index": "testA",  
  "value": "FLEXISPOT"
}},
{{
  "index": "testB",  
  "value": "4.7"
}}
]

The context below.\n
------------\n
{context_str}\n
------------\n

Analyzing the entire passage, the strengths and weaknesses of this product (positive and negative reviews) can be summarized.

answer the question:: {question} \n
"""

PROS_CONS_QUESTION_PROMPT = PromptTemplate(
    template=pros_cons_prompt_template, input_variables=["question", "context_str"]
)


pros_cons_refine_prompt_template = """Use the following pieces of context to answer the question at the end. 
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.

The original question is the list of attributes which want to be queried.like this: ["testA","testB"]

Do not include special characters, such as ",' and so on in the values of the answer.
That may cause the json format to be invalid.Your answer should be in json format. Don't include other words in your answer.
The words of your answer should be in English.
then your output format must be valid json format,it will be like this:

[{{
  "index": "testA",  
  "value": "FLEXISPOT"
}},
{{
  "index": "testB",  
  "value": "4.7"
}}
]

The original question is as follows: {question} \n

We have provided an existing answer, including sources: {existing_answer}\n
We have the opportunity to refine the existing answer
(only if needed) with some more context below.\n
------------\n
{context_str}\n
------------\n

The original question is in Chinese, but the context is in English.
You can translate the question to English if you want to.

Analyzing the entire passage, the strengths and weaknesses of this product (positive and negative reviews) can be summarized.
Your answer should be in English.

Given the new context, refine the original answer to better 
answer the question. 
If you do update it, please update the sources as well.
If the context isn't useful, return the original answer.
"""

PROS_CONS_REFINE_PROMPT = PromptTemplate(
    template=pros_cons_refine_prompt_template, input_variables=["question", "existing_answer", "context_str"]
)
chain_type_kwargs = {"question_prompt":PROS_CONS_QUESTION_PROMPT,"refine_prompt": PROS_CONS_REFINE_PROMPT}
prosConsQaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="refine", 
                                         retriever=WebContent2LocalFileSplitRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

def get_pros_cons_disadvantages(index_contents:[str]) -> str:
    try:
        result = prosConsQaChain(index_contents)
        print(result["answer"])
        return result["answer"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""