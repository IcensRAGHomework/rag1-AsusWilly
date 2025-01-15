import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def create_openai_model():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

llm = create_openai_model()

def format_json(data):
    return json.dumps(data, indent=4, ensure_ascii=False)

def generate_hw01(question):
    llm.bind(response_format={"type": "json_object"})
    format_instructions = '{{"Result": [{{ "date": "yyyy-MM-dd", "name": "節日" }}, {{ "date": "yyyy-MM-dd", "name": "節日" }}] }}'

    response = llm.invoke([question + f"{format_instructions}"])
    json_output = format_json(JsonOutputParser().invoke(response))
    return json_output
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

query = "2024年台灣10月紀念日有哪些?"
print(generate_hw01(query))