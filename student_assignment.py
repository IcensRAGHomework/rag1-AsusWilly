import json
import requests

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

def get_holiday_from_calendarific(year, month, country):

    API_KEY = "w9O94T1c8qE1Y2m3jgStYH27d1HgS0rF"
    BASE_URL = "https://calendarific.com/api/v2/holidays"

    params = {
        "api_key": API_KEY,
        "year": year,
        "month": month,
        "country": country
    }
    response = requests.get(BASE_URL, params=params)
    # 檢查 API 響應狀態
    if response.status_code == 200:
        data = response.json()
        holidays = data.get("response", {}).get("holidays", [])

        # 篩選符合條件的節日（例如 Observance: 紀念日）
        result = []
        for holiday in holidays:
            result.append({
                "date": holiday["date"]["iso"],
                "name": holiday["name"]
             })
        return {"Result": result}
    else:
        # 若出錯，返回錯誤訊息
        return {"Error": f"API Error: {response.status_code}, {response.text}"}

def generate_hw01(question):
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

print(get_holiday_from_calendarific(2024, 10, "TW"))
#query = "2024年台灣10月紀念日有哪些?"
#print(generate_hw01(query))