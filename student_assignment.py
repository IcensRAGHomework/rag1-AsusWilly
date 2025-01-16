import json
import re
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

def get_holidays_from_calendarific(year, month, country_code):

    API_KEY = "w9O94T1c8qE1Y2m3jgStYH27d1HgS0rF"
    BASE_URL = "https://calendarific.com/api/v2/holidays"

    params = {
        "api_key": API_KEY,
        "year": year,
        "month": month,
        "country": "TW",
        "language": "zh"
    }
    return requests.get(BASE_URL, params=params)
    
def extract_info_from_question(question):
    response = llm.invoke([
        {"role": "system", "content": "你是一個專門提取問題資訊的助理，返回結構化的 JSON 格式，包含年份、月份和國家。"},
        {"role": "user", "content": f"請根據以下問題提取資訊，並返回 JSON 格式，包含三個鍵值：\n"
                                    f"- 'year': 問題中的年份（若無，填寫 null）\n"
                                    f"- 'month': 問題中的月份（若無，填寫 null）\n"
                                    f"- 'country': 問題中的國家代碼(ISO 3166-1 alpha-2)\n\n"
                                    f"問題: '{question}'"}
    ])
    
    try:
        # 使用正则表达式提取年份、月份和国家
        json_match = re.search(r"{.*}", response.content, re.DOTALL)
        if json_match:
            clean_json = json_match.group()
            parsed_result = json.loads(clean_json)

            # 创建返回的 JSON 格式
            result = {
                "year": parsed_result.get("year", None),
                "month": parsed_result.get("month", None),
                "country": parsed_result.get("country", None)
            }
            return result

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_holiday_info(question):
    # 提取年份、月份和国家
    extracted_info = extract_info_from_question(question)

    if extracted_info and all(extracted_info[key] is not None for key in ['year', 'month', 'country']):
        # 获取节假日数据
        year = extracted_info['year']
        month = extracted_info['month']
        country = extracted_info['country']

        # 调用 Calendarific API 获取节假日
        holidays_data = get_holidays_from_calendarific(year, month, country).json()
        
        # 提取节假日名称
        #holidays = [holiday["name"] for holiday in holidays_data["response"]["holidays"]]
        return {
            "Result": [
                {
                    "date": holiday["date"]["iso"],
                    "name": holiday["name"]
                }
                for holiday in holidays_data["response"]["holidays"]
            ]
        }
    else:
        return "无法从问题中提取到有效的年份、月份或国家。"

def generate_hw01(question):
    format_instructions = '{{"Result": [{{ "date": "yyyy-MM-dd", "name": "節日" }}, {{ "date": "yyyy-MM-dd", "name": "節日" }}] }}'

    response = llm.invoke([question + f"{format_instructions}"])
    json_output = format_json(JsonOutputParser().invoke(response))
    return json_output
    
def generate_hw02(question):
    return get_holiday_info(question)
    # response = llm.invoke([
    #     {"role": "system", "content": "你是一個專門提取問題資訊的助理，只返回符合要求的 JSON 格式，不包含其他文字或說明。"},
    #     {"role": "user", "content": f"請根據以下問題提取資訊，並返回 JSON 格式，包含以下三個鍵值：\n"
    #                                 f"- 'year': 問題中的年份（若無，填寫 null）\n"
    #                                 f"- 'month': 問題中的月份（若無，填寫 null）\n"
    #                                 f"- 'country': 問題中的國家或地區名稱（若無，填寫 null）\n\n"
    #                                 f"問題: '{question}'"}])
    # raw_content = response.content.strip()
    # print(f"Debug: Response content: {raw_content}")  # 调试输出
    
    # json_match = re.search(r"{.*}", raw_content, re.DOTALL)
    # if json_match:
    #     clean_json = json_match.group()
    #     try:
    #         parsed_result = json.loads(clean_json)
    #         return get_holiday_from_calendarific(parsed_result.get("year", None), parsed_result.get("month", None), "TW")
    #     except json.JSONDecodeError as e:
    #         print(f"Error parsing JSON: {e}")
    #         return None
    # else:
    #     print("Error: 返回的内容不包含有效的 JSON 格式。")
    #     return None
    
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

# print(get_holiday_from_calendarific(2024, 10, "TW"))
# query = "2024年台灣10月紀念日有哪些?"
# print(generate_hw01(query))
query = "2024年台灣10月紀念日有哪些?"
print(generate_hw02(query))