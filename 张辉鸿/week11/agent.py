import asyncio
import os
import uuid
from agents import Agent, Runner, set_default_openai_api
from agents.run import RunResultStreaming
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent


os.environ["OPENAI_API_KEY"] = "sk-642d5cd9c606477badc2e08919f6fa2c"  # 替换为你的 Key
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 设置使用 chat_completions API (阿里云DashScope兼容模式需要)
set_default_openai_api("chat_completions")


agent1 = Agent(
    name="Sentiment Classifier",
    model="qwen-max",
    handoff_description="负责处理所有关于【情感分类】、【情绪判断】的任务。",
    instructions="""你是一个情感分析专家。
    请判断用户输入的文本是正面、负面还是中性情绪，并用简短的一句话解释原因。
    不要输出任何多余的废话。"""
)


# 2. 定义子agent实体识别专家
agent2 = Agent(
    name="Entity Recognizer",
    model="qwen-max",
    handoff_description="负责处理所有关于【实体识别】、【信息提取】（如人名、地名、组织机构、时间等）的任务。",
    instructions="""你是一个命名实体识别(NER)专家。
    请精准提取文本中的关键实体，并以清晰的 Markdown 列表格式输出，如：
    - 人名：XXX
    - 地点：XXX
    - 组织：XXX"""
)

# 3. 定义主 Agent：任务调度与汇总 (Orchestrator)
main_agent = Agent(
    name="Task Orchestrator",
    model="qwen-max",
    instructions="""你是任务调度主控管家。你的职责是：
    1. 接收用户的文本处理请求。
    2. 判断请求类型，并将任务准确转交（Handoff）给最合适的专家代理。
    3. 拿到专家代理的结果后，整理并输出给用户。""",
    handoffs=[agent1, agent2]
)



# 4. 执行多智能体测试

async def main():

    while True:
        msg = input("请输入: ")
        
        if not msg.strip():
            continue
        
        try:
            result = await Runner.run(main_agent, msg)
            print(f"处理结果:\n{result.final_output}\n")
        except Exception as e:
            print(f"\n发生错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())