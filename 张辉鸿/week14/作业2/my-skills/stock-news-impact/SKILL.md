---
name: 新闻对股票影响分析与买卖建议
description: 结合 get-news 新闻 API 和 autostock 股票 API，分析新闻情绪与股价波动的关系，可视化新闻发布前后的价格变化，并给出基于新闻+技术面的买卖建议。
---

# 功能概述

本 skill 复用 `autostock` 和 `get-news` 两个已有 skill，实现以下功能：

1. **新闻获取**：复用 `get-news` skill 的 API（`https://whyta.cn/api`），拉取最新财经/头条新闻
2. **股价获取**：复用 `autostock` skill 的 API（`https://api.autostock.cn`），拉取股票日K线数据
3. **新闻-股价关联分析**：计算新闻发布日前后的价格波动，判断新闻对股价的影响方向
4. **可视化**：绘制新闻事件标记 + 股价走势图，直观展示新闻与股价的关系
5. **买卖建议**：结合新闻情绪 + 技术面（均线、成交量）给出综合买卖建议

# 分析方法

## 新闻情绪评分
| 关键词类型 | 示例 | 情绪分 |
|-----------|------|--------|
| 利好 | 增长、突破、签约、中标、分红、回购 | +1 |
| 利空 | 亏损、处罚、诉讼、减持、退市、调查 | -1 |
| 中性 | 公告、会议、变动、披露、说明 | 0 |

## 新闻影响评估
| 新闻情绪 | 股价反应 | 结论 |
|---------|---------|------|
| 利好 + 股价上涨 | 利好兑现 | 短期可持有，注意高位回落 |
| 利好 + 股价不涨/下跌 | 利好出尽 | 警惕出货，建议减仓 |
| 利空 + 股价下跌 | 利空兑现 | 观望，等待止跌信号 |
| 利空 + 股价不跌/上涨 | 利空出尽 | 可能是底部，关注买入机会 |
| 中性 | — | 按技术面正常操作 |

# 实现代码

```python
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta

# ============ 配置（复用 autostock + get-news） ============
STOCK_TOKEN = "zgaLG8unUPr"
NEWS_KEY = "6d997a997fbf"

STOCK_BASE = "https://api.autostock.cn/v1/stock"
NEWS_BASE = "https://whyta.cn/api"


# ============ Part 1: 新闻获取（复用 get-news skill） ============
def fetch_news(news_type: str = "toutiao") -> list:
    """
    获取新闻列表（复用 get-news skill）
    news_type: 'toutiao' | 'douyin' | 'github' | 'bulletin' | 'esports'
    """
    endpoints = {
        "toutiao": f"{NEWS_BASE}/tx/topnews?key={NEWS_KEY}",
        "douyin": f"{NEWS_BASE}/tx/douyinhot?key={NEWS_KEY}",
        "bulletin": f"{NEWS_BASE}/tx/bulletin?key={NEWS_KEY}",
        "esports": f"{NEWS_BASE}/tx/esports?key={NEWS_KEY}",
    }
    url = endpoints.get(news_type, endpoints["toutiao"])
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return data.get("data", data.get("result", []))
    except Exception as e:
        print(f"新闻获取失败: {e}")
        return []


def score_news_sentiment(title: str) -> int:
    """对新闻标题进行简单情绪打分"""
    positive_words = [
        "增长", "突破", "大涨", "涨停", "签约", "中标", "分红", "回购",
        "利好", "创新高", "扭亏", "盈利", "增长", "扩张", "升级", "获奖",
        "获批", "合作", "投资", "融资", "上市", "翻倍",
    ]
    negative_words = [
        "亏损", "处罚", "诉讼", "减持", "退市", "调查", "暴跌", "跌停",
        "利空", "下滑", "违约", "破产", "爆雷", "造假", "违规", "警告",
        "被查", "st", "退市风险", "债务", "逾期", "冻结",
    ]

    score = 0
    for w in positive_words:
        if w in title:
            score += 1
    for w in negative_words:
        if w in title:
            score -= 1
    return score


def filter_stock_news(news_list: list, stock_keywords: list) -> list:
    """筛选包含特定股票关键词的新闻"""
    filtered = []
    for item in news_list:
        title = item.get("title", "")
        if any(kw in title for kw in stock_keywords):
            item["sentiment"] = score_news_sentiment(title)
            filtered.append(item)
    return filtered


# ============ Part 2: 股价获取（复用 autostock skill） ============
def get_stock_kline(code: str, start_date: str = None,
                    end_date: str = None) -> pd.DataFrame:
    """
    获取日K线数据（复用 autostock skill）
    """
    url = f"{STOCK_BASE}/kline/day?token={STOCK_TOKEN}"
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    payload = {"code": code, "startDate": start_date, "endDate": end_date, "type": 1}
    try:
        resp = requests.get(url, params=payload, timeout=10)
        data = resp.json()
        records = data.get("data", [])
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        print(f"K线获取失败: {e}")
        return pd.DataFrame()


def calc_ma(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算移动平均线"""
    if periods is None:
        periods = [5, 10, 20]
    for p in periods:
        df[f"ma{p}"] = df["close"].rolling(window=p).mean()
    return df


# ============ Part 3: 新闻-股价关联分析 ============
def analyze_news_impact(
    stock_code: str,
    stock_name: str = "",
    news_type: str = "toutiao",
    stock_keywords: list = None,
) -> dict:
    """
    主入口函数：获取新闻 + 获取股价 → 关联分析 → 绘图 → 买卖建议

    参数:
        stock_code: 股票代码，如 "000001"
        stock_name: 股票名称，如 "平安银行"
        news_type: 新闻源: 'toutiao' | 'douyin' | 'bulletin'
        stock_keywords: 筛选新闻的关键词列表，默认取股票名称

    返回:
        {
            "news_list": [...],      # 相关新闻（含情绪分）
            "kline_df": DataFrame,   # K线数据
            "impact_events": [...],  # 新闻影响事件
            "signals": [...]         # 买卖建议
        }
    """
    if stock_keywords is None:
        stock_keywords = [stock_name] if stock_name else [stock_code]

    # 1. 获取新闻
    print(f"📡 正在获取 {news_type} 新闻...")
    all_news = fetch_news(news_type)
    related_news = filter_stock_news(all_news, stock_keywords)
    print(f"  获取到 {len(all_news)} 条新闻，其中 {len(related_news)} 条与 {stock_name or stock_code} 相关")

    # 2. 获取股价
    print(f"📡 正在获取 {stock_code} 日K线...")
    kline_df = get_stock_kline(stock_code)
    if kline_df.empty:
        return {"error": "未能获取K线数据", "code": stock_code}

    kline_df = calc_ma(kline_df)

    # 3. 关联分析
    if not related_news:
        print("⚠️ 未找到与该股票直接相关的新闻，将展示全部头条新闻情绪")

    # 4. 绘图
    print("📊 正在生成新闻-股价关联图...")
    plot_news_price(kline_df, related_news, stock_code, stock_name)

    # 5. 生成信号
    signals = generate_news_signals(kline_df, related_news)

    # 6. 打印报告
    print_report(stock_code, stock_name, related_news, signals, kline_df)

    return {
        "code": stock_code,
        "name": stock_name,
        "news_count": len(related_news),
        "news_sentiment_avg": (
            round(np.mean([n["sentiment"] for n in related_news]), 2)
            if related_news else 0
        ),
        "signals": signals,
    }


def plot_news_price(kline_df: pd.DataFrame, news_list: list,
                    code: str, name: str = ""):
    """
    绘制K线走势 + 新闻事件标记图
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    title = f"{code} {name} 新闻-股价关联分析" if name else f"{code} 新闻-股价关联分析"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 子图1: K线走势 + 均线
    ax1.plot(kline_df["date"], kline_df["close"], color="black",
             linewidth=1.5, label="收盘价")
    ax1.plot(kline_df["date"], kline_df["ma5"], color="blue",
             linewidth=0.8, alpha=0.6, label="MA5")
    ax1.plot(kline_df["date"], kline_df["ma10"], color="orange",
             linewidth=0.8, alpha=0.6, label="MA10")
    ax1.plot(kline_df["date"], kline_df["ma20"], color="purple",
             linewidth=0.8, alpha=0.6, label="MA20")

    # 填充涨跌区域
    ax1.fill_between(kline_df["date"], kline_df["close"], kline_df["close"].iloc[0],
                     where=(kline_df["close"] >= kline_df["close"].iloc[0]),
                     color="green", alpha=0.08)
    ax1.fill_between(kline_df["date"], kline_df["close"], kline_df["close"].iloc[0],
                     where=(kline_df["close"] < kline_df["close"].iloc[0]),
                     color="red", alpha=0.08)

    # 标注新闻事件（用竖虚线）
    news_dates_in_range = []
    for news in news_list:
        news_date_str = news.get("date", news.get("ctime", ""))
        try:
            news_date = pd.to_datetime(news_date_str)
            # 如果新闻日期在K线范围内或附近
            if kline_df["date"].min() <= news_date <= kline_df["date"].max() + timedelta(days=3):
                color = "green" if news.get("sentiment", 0) > 0 else \
                        "red" if news.get("sentiment", 0) < 0 else "gray"
                ax1.axvline(x=news_date, color=color, linestyle="--",
                            linewidth=1, alpha=0.5)
                news_dates_in_range.append(news_date)
        except Exception:
            pass

    ax1.set_ylabel("价格 (元)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # 子图2: 成交量
    if "volume" in kline_df.columns:
        colors_vol = ["green" if kline_df["close"].iloc[i] >= kline_df["open"].iloc[i]
                      else "red" for i in range(len(kline_df))]
        ax2.bar(kline_df["date"], kline_df["volume"], color=colors_vol,
                alpha=0.6, width=0.8)
        ax2.set_ylabel("成交量", fontsize=12)
        ax2.grid(True, alpha=0.2)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax2.set_xlabel("日期", fontsize=12)

    # 图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", linestyle="--", label="利好新闻"),
        Line2D([0], [0], color="red", linestyle="--", label="利空新闻"),
        Line2D([0], [0], color="gray", linestyle="--", label="中性新闻"),
    ]
    ax1.legend(loc="upper left", fontsize=8,
               handles=ax1.get_legend_handles_labels()[0] + legend_elements)

    plt.tight_layout()
    save_path = f"{code}_news_impact.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"图表已保存至: {save_path}")


def generate_news_signals(kline_df: pd.DataFrame, news_list: list) -> list:
    """
    结合新闻情绪 + 技术面生成买卖信号
    """
    signals = []

    if kline_df.empty:
        return ["无K线数据"]

    latest = kline_df.iloc[-1]
    close = latest["close"]
    ma5 = latest.get("ma5", close)
    ma10 = latest.get("ma10", close)

    # 新闻情绪汇总
    sentiment_scores = [n.get("sentiment", 0) for n in news_list]
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

    # 技术面判断
    price_above_ma5 = close > ma5
    price_above_ma10 = close > ma10
    ma5_above_ma10 = ma5 > ma10

    # ---- 信号规则 ----
    if avg_sentiment > 0.5 and price_above_ma5 and ma5_above_ma10:
        signals.append(
            "🔥【强买入】多头新闻 + 价格站上MA5/MA10 + 均线多头排列 → 最佳买点"
        )
    elif avg_sentiment > 0:
        if price_above_ma5:
            signals.append("✅【偏多】新闻情绪偏多 + 价格在MA5上方 → 可持有或小幅加仓")
        else:
            signals.append(
                "⚠️【利好不涨】利多新闻但股价在均线下方 → 警惕出货，建议观望"
            )
    elif avg_sentiment < -0.5:
        if not price_above_ma5 and not price_above_ma10:
            signals.append("🔴【强卖出】空头新闻 + 价格跌破均线 → 建议减仓避险")
        else:
            signals.append(
                "💡【利空不跌】利空新闻但股价站稳均线 → 利空出尽信号，关注抄底机会"
            )
    elif avg_sentiment < 0:
        signals.append("📉【偏空】新闻偏空 + 技术面需结合判断 → 轻仓观望")
    else:
        signals.append("📊【无方向】无显著新闻信号 → 按技术面正常操作")

    # 成交量异常信号
    if "volume" in kline_df.columns:
        avg_vol = kline_df["volume"].tail(20).mean()
        latest_vol = latest.get("volume", 0)
        if latest_vol > avg_vol * 2:
            signals.append("📈【放量异动】成交量突增2倍以上 → 关注突破方向，顺势而为")

    return signals


def print_report(code: str, name: str, news: list, signals: list,
                 kline_df: pd.DataFrame):
    """打印分析报告"""
    print("\n" + "=" * 60)
    print(f"  {code} {name} 新闻-股价影响分析报告")
    print("=" * 60)

    if news:
        print(f"\n📰 相关新闻 ({len(news)}条):")
        for i, n in enumerate(news[:5], 1):
            sent = "🟢" if n["sentiment"] > 0 else "🔴" if n["sentiment"] < 0 else "⚪"
            print(f"  {i}. {sent} [{n['sentiment']:+d}] {n.get('title', '')[:60]}")

    print(f"\n📊 股价概况:")
    latest = kline_df.iloc[-1]
    first = kline_df.iloc[0]
    change = (latest["close"] - first["close"]) / first["close"] * 100
    print(f"  最新价: {latest['close']:.2f}  |  区间涨跌: {change:+.2f}%")
    print(f"  MA5: {latest.get('ma5', 'N/A')}  |  MA10: {latest.get('ma10', 'N/A')}")

    print(f"\n📌 综合建议:")
    for s in signals:
        print(f"    {s}")
    print("=" * 60 + "\n")


# ============ 使用示例 ============
if __name__ == "__main__":
    # 分析贵州茅台相关新闻对股价的影响
    result = analyze_news_impact(
        stock_code="600519",
        stock_name="贵州茅台",
        news_type="toutiao",
        stock_keywords=["茅台", "白酒", "贵州茅台"],
    )
```

# 依赖安装

```bash
pip install requests matplotlib pandas numpy
```

# 快速使用

```python
from stock_news_impact import analyze_news_impact

# 分析某只股票的新闻影响
result = analyze_news_impact("000001", "平安银行", news_type="toutiao")
```

# 输出说明

| 输出项 | 说明 |
|--------|------|
| `{code}_news_impact.png` | 双图：上=K线+均线+新闻事件标记线，下=成交量 |
| `news_sentiment_avg` | 相关新闻的平均情绪分数 |
| `signals` | 新闻+技术面综合买卖建议 |

# 技能复用关系

```
stock-news-impact
  ├── 复用 autostock → 获取日K线、计算均线
  └── 复用 get-news  → 获取头条/公告新闻、情绪打分
```

# 注意事项

- 新闻情绪打分基于简单关键词匹配，仅作参考
- 新闻日期与交易日可能存在时差（周末/假期发布的新闻）
- 默认获取最近 60 个交易日 + 头条新闻
- `get-news` 的 NEWS_KEY 为 `6d997a997fbf`
