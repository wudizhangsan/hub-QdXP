from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal
import openai
import json

client = openai.OpenAI(
    api_key="sk-642d5cd9c606477badc2e08919f6fa2c", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Translation(BaseModel):
    """文本翻译请求"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    translated_text: str = Field(description="将原始文本翻译为目标语种后的最终结果")

result = ExtractionAgent(model_name="qwen-plus").call('将World Peace！翻译为中文', Translation)
print(result)

result = ExtractionAgent(model_name="qwen-plus").call('你好，世界！翻译为英文', Translation)
print(result)
