import json
import re
import requests

from model_configurations import get_model_configuration

from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.schema.runnable import RunnableLambda

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
format_instructions = '{{"Result": [{{ "date": "yyyy-MM-dd", "name": "節日" }}, {{ "date": "yyyy-MM-dd", "name": "節日" }}] }}'

def format_json(data):
    formatted = json.dumps(data, indent=4, ensure_ascii=False)
    formatted = formatted.replace('"Result": {', '"Result":\n        {')
    formatted = formatted.replace('\n    }', '\n        }')
    return formatted

def get_holidays_from_calendarific(year, month, country_code):

    API_KEY = 'w9O94T1c8qE1Y2m3jgStYH27d1HgS0rF'
    BASE_URL = 'https://calendarific.com/api/v2/holidays'

    params = {
        'api_key': API_KEY,
        'year': year,
        'month': month,
        'country': country_code
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f'API request failed with status {response.status_code}'}

@tool
def holidays_tool(country_code, year, month):
    '獲取某國家某年某月的紀念日。country_code 為國家代碼(ISO 3166-1 alpha-2)。year 為年。month 為月'
    holidays_data = get_holidays_from_calendarific(year, month, country_code)

    if 'error' in holidays_data:
        return holidays_data['error']

    result = [
        {
            'date': holiday['date']['iso'],
            'name': holiday['name']
        }
        for holiday in holidays_data.get('response', {}).get('holidays', [])
    ]

    return result
    
def get_holiday_info_with_agent(question):
    prompt_template = PromptTemplate(
        input_variables = ['question', 'agent_scratchpad'],
        template = '{question}' + format_instructions + '\nAgent Scratchpad: {agent_scratchpad}'
    )
    agent = create_tool_calling_agent(llm, [holidays_tool], prompt_template)
    response = AgentExecutor(
        agent= agent,
        tools = [holidays_tool],
        verbose =False
    ).invoke({'question': question}) 
    return response

def generate_hw01(question):
    response = llm.invoke([question + f'{format_instructions}'])
    return format_json(JsonOutputParser().invoke(response))
    
def generate_hw02(question):
    response = get_holiday_info_with_agent(question)
    return format_json(JsonOutputParser().invoke(response['output']))

session_memories = {}

def get_session_history(session_id):
    if session_id not in session_memories:
        session_memories[session_id] = InMemoryChatMessageHistory()
    return session_memories[session_id]

def generate_hw03(question2, question3):
    holiday_agent_runnable = RunnableLambda(get_holiday_info_with_agent)
    agent_with_memory = RunnableWithMessageHistory(
        runnable = holiday_agent_runnable,
        get_session_history = get_session_history
    )
    session_id = 'user-session' 
    response = agent_with_memory.invoke(
        {'question': question2 + f'{format_instructions}'},
        config={'session_id': session_id} 
    )

    response = agent_with_memory.invoke(
        {'question': question3 + '是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，'
        + '如果不存在，則為 true；否則為 false，並描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，'
        + '以及當前清單的內容。答案請用此 JSON 格式呈現:{{ "Result": {{ "add": true, "reason": "描述" }} }}'},
        config={'session_id': session_id} 
    )
    return format_json(JsonOutputParser().invoke(response['output']))
    
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

#print(generate_hw01('2024年台灣10月紀念日有哪些?'))
#print(generate_hw02('2024年台灣10月紀念日有哪些?'))
print(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單?'))
