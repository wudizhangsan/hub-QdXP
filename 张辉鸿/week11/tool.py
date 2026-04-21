import re
from typing import Annotated, Union
import requests
import random
import string


TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"


@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


# ==========================================
# 新增工具 1：安全密码生成器
# ==========================================
@mcp.tool
def generate_secure_password(
        length: Annotated[int, "生成密码的长度，默认建议为12位以上"] = 12
) -> str:
    """生成一个包含大小写字母、数字和特殊字符的随机高强度安全密码。"""
    if length < 8:
        length = 8
    # 定义密码字符集
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    # 随机生成
    secure_pwd = "".join(random.choice(characters) for _ in range(length))
    return f"生成的 {length} 位安全密码为: {secure_pwd}"


# ==========================================
# 新增工具 2：汇率换算器 (Mock数据)
# ==========================================
@mcp.tool
def calculate_exchange_rate(
        amount: Annotated[float, "需要转换的金钱数额"],
        from_currency: Annotated[str, "原货币代码，例如: USD, CNY, EUR"],
        to_currency: Annotated[str, "目标货币代码，例如: CNY, USD, EUR"]
) -> str:
    """计算两种货币之间的汇率转换结果。支持的货币有：USD(美元), CNY(人民币), EUR(欧元), JPY(日元), GBP(英镑)。"""
    # 模拟相对美元的汇率库
    mock_rates = {"USD": 1.0, "CNY": 7.25, "EUR": 0.92, "JPY": 150.5, "GBP": 0.79}

    from_c = from_currency.upper()
    to_c = to_currency.upper()

    if from_c not in mock_rates or to_c not in mock_rates:
        return "换算失败：不支持该货币类型。目前仅支持 USD, CNY, EUR, JPY, GBP。"

    # 先折算成美元，再折算成目标货币
    usd_amount = amount / mock_rates[from_c]
    target_amount = usd_amount * mock_rates[to_c]

    return f"【汇率换算结果】 {amount} {from_c} 约等于 {target_amount:.2f} {to_c}。"

# ==========================================
# 新增工具 3：BMI 健康指数计算器
# ==========================================
@mcp.tool
def calculate_bmi(
        height_cm: Annotated[float, "用户的身高，单位：厘米"],
        weight_kg: Annotated[float, "用户的体重，单位：千克"]
) -> str:
    """根据身高和体重计算 BMI 指数，并返回健康状态评估。"""
    height_m = height_cm / 100.0  # 厘米转米
    bmi = weight_kg / (height_m * height_m)

    if bmi < 18.5:
        status = "偏瘦，建议多吃点！"
    elif 18.5 <= bmi < 24:
        status = "正常，非常健康！"
    elif 24 <= bmi < 28:
        status = "微胖，建议控制饮食哦。"
    else:
        status = "肥胖，要注意加强锻炼啦！"

    return f"计算完成：您的BMI指数为 {bmi:.2f}。体型评估：{status}"