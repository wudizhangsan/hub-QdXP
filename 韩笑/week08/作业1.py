from pydantic import BaseModel, Field


import openai


client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现信息信息抽取
指定写的tool的格式
"""
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
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get('title', 'Extraction'),
                    "description": schema.get('description', 'Extract information'),
                    "parameters": {
                        "type": "object",
                        "properties": schema.get('properties', {}),
                        "required": schema.get('required', []),
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
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print('ERROR', e)
            print('Raw Message:', response.choices[0].message)
            return None

#"识别翻译任务的关键信息：源语言、目标语言和待翻译文本
class Translation(BaseModel):

    source_lang: str = Field(description="原始语种，例如：英文、中文、法语")
    target_lang: str = Field(description="目标语种，例如：中文、英文、日文")
    text: str = Field(description="需要被翻译的原始文本内容")

result = ExtractionAgent(model_name="qwen-plus").call("帮我将 good！翻译为中文", Translation)
print(result)
if result:
    print(f"原始语种: {result.source_lang}")
    print(f"目标语种: {result.target_lang}")
    print(f"待翻译文本: {result.text}")
