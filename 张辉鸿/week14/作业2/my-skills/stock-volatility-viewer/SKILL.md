---
name: 股票波动率可视化与买卖建议
description: 基于 autostock API 获取股票日K/周K数据，计算日波动率和周波动率并绘制在同一张图中，同时根据波动大小给出最佳买卖时机建议。
---

# 功能概述

本 skill 实现以下核心功能：
1. **获取数据**：复用 `autostock` skill 的 API（`https://api.autostock.cn`），拉取指定股票的日K线和周K线数据
2. **波动率计算**：日波动率 = (当日最高 - 当日最低) / 当日收盘价 × 100%；周波动率同理
3. **双图可视化**：将日波动率（柱状图）和周波动率（柱状图）上下排列在同一张图中，标注均值参考线
4. **买卖建议**：基于波动大小、波动趋势和价格方向给出最佳买入/卖出时机建议

# 波动率分析理论

## 波动率公式
```
日波动率 = (high - low) / close × 100%
周波动率 = (本周最高 - 本周最低) / 本周收盘价 × 100%
```

## 波动率大小含义
| 波动状态 | 阈值 | 含义 |
|---------|------|------|
| 极度高波动 | > 均值 + 2σ | 恐慌或狂热，即将反转 |
| 偏高波动 | 均值 + 1σ ~ 均值 + 2σ | 趋势加速，顺势操作 |
| 正常波动 | 均值 ± 1σ | 正常行情，按趋势判断 |
| 低波动盘整 | < 均值 - 1σ | 酝酿突破，等待方向 |

## 买卖信号规则
| 条件 | 信号 | 操作建议 |
|------|------|---------|
| 日高波动 + 阳线收盘 | ✅ 强势买入 | 顺势做多，设止损于当日低点 |
| 日高波动 + 阴线收盘 | ⚠️ 卖出/减仓 | 减仓或清仓观望 |
| 日低波动盘整 | 📊 等待 | 等待放量突破方向 |
| 周高波动 > 均值1.5倍 | 📈 中期信号 | 中期趋势启动，重点关注 |
| 日波动 + 周波动同时放大 | 🔥 共振信号 | 最佳买卖时机，重仓操作 |
| 日波动骤降（缩量） | ⏸️ 休息 | 市场冷清，不宜入场 |

# 实现代码

```python
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============ 配置（复用 autostock skill） ============
TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1/stock/kline"


def get_kline_data(code: str, kline_type: str, start_date: str = None,
                   end_date: str = None, fq_type: int = 1) -> dict:
    """
    获取K线数据（复用 autostock API）
    kline_type: 'day' | 'week' | 'month'
    fq_type: 0=不复权, 1=前复权, 2=后复权
    """
    url = f"{BASE_URL}/{kline_type}?token={TOKEN}"
    payload = {
        "code": code,
        "startDate": start_date,
        "endDate": end_date,
        "type": fq_type,
    }
    resp = requests.get(url, params=payload, timeout=10)
    return resp.json()


def calc_volatility_df(raw_data: list) -> pd.DataFrame:
    """
    将原始K线数据转为DataFrame并计算波动率
    波动率 = (最高价 - 最低价) / 收盘价 × 100
    """
    df = pd.DataFrame(raw_data)
    df["date"] = pd.to_datetime(df["date"])
    df["volatility"] = (df["high"] - df["low"]) / df["close"] * 100
    return df.sort_values("date")


def plot_dual_volatility(day_df: pd.DataFrame, week_df: pd.DataFrame,
                         code: str, name: str = ""):
    """
    核心绘图函数：日波动 + 周波动上下排列在同一张图中
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=False,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    title = f"{code} {name} 波动率分析" if name else f"{code} 波动率分析"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # ---- 日波动率 ----
    colors_day = ["#2ecc71" if row["close"] >= row["open"] else "#e74c3c"
                  for _, row in day_df.iterrows()]
    ax1.bar(day_df["date"], day_df["volatility"], color=colors_day,
            alpha=0.75, width=0.8, label="日波动率(%)")
    d_mean = day_df["volatility"].mean()
    d_std = day_df["volatility"].std()
    ax1.axhline(y=d_mean, color="blue", linestyle="-", linewidth=1.2,
                label=f"均值: {d_mean:.2f}%")
    ax1.axhline(y=d_mean + 2 * d_std, color="red", linestyle="--",
                linewidth=0.8, alpha=0.6, label=f"+2σ: {d_mean + 2*d_std:.2f}%")
    ax1.axhline(y=d_mean - d_std, color="gray", linestyle="--",
                linewidth=0.8, alpha=0.6, label=f"-1σ: {d_mean - d_std:.2f}%")
    ax1.set_ylabel("日波动率 (%)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # 高波动阈值标注（日线）
    extreme_days = day_df[day_df["volatility"] > d_mean + 2 * d_std]
    for _, row in extreme_days.iterrows():
        ax1.annotate(f'{row["volatility"]:.1f}%', (row["date"], row["volatility"]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7,
                     color="red", ha="center")

    # ---- 周波动率 ----
    colors_week = ["#2ecc71" if row["close"] >= row["open"] else "#e74c3c"
                   for _, row in week_df.iterrows()]
    ax2.bar(week_df["date"], week_df["volatility"], color=colors_week,
            alpha=0.75, width=3, label="周波动率(%)")
    w_mean = week_df["volatility"].mean()
    ax2.axhline(y=w_mean, color="blue", linestyle="-", linewidth=1.2,
                label=f"均值: {w_mean:.2f}%")
    ax2.axhline(y=w_mean * 1.5, color="orange", linestyle="--",
                linewidth=0.8, alpha=0.6, label=f"1.5×均值: {w_mean*1.5:.2f}%")
    ax2.set_xlabel("日期", fontsize=12)
    ax2.set_ylabel("周波动率 (%)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    # 高波动阈值标注（周线）
    extreme_weeks = week_df[week_df["volatility"] > w_mean * 1.5]
    for _, row in extreme_weeks.iterrows():
        ax2.annotate(f'{row["volatility"]:.1f}%', (row["date"], row["volatility"]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7,
                     color="red", ha="center")

    plt.tight_layout()
    save_path = f"{code}_volatility.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"图表已保存至: {save_path}")
    return fig


def generate_signals(day_df: pd.DataFrame, week_df: pd.DataFrame) -> list:
    """
    基于波动大小生成买卖信号

    信号优先级（从高到低）：
    1. 日周共振高波动  → 最强信号
    2. 日线极端波动    → 短期反转或加速
    3. 周线异常波动    → 中期趋势变化
    4. 低波动盘整      → 等待突破
    """
    day_vol = day_df["volatility"]
    week_vol = week_df["volatility"]

    d_mean = day_vol.mean()
    d_std = day_vol.std()
    w_mean = week_vol.mean()

    latest_day = day_df.iloc[-1]
    latest_week = week_df.iloc[-1]

    latest_day_vol = latest_day["volatility"]
    latest_week_vol = latest_week["volatility"]

    day_up = latest_day["close"] >= latest_day["open"]  # 阳线
    week_up = latest_week["close"] >= latest_week["open"]

    signals = []
    score = 0  # 综合评分，正=偏买，负=偏卖

    # ---- 规则1: 日周共振 ----
    if latest_day_vol > d_mean + d_std and latest_week_vol > w_mean * 1.5:
        if day_up and week_up:
            signals.append("🔥【共振买入】日线+周线同时高波动且收阳 — 最佳买点！建议重仓，止损设在周最低价")
            score += 3
        elif not day_up and not week_up:
            signals.append("🔴【共振卖出】日线+周线同时高波动且收阴 — 最佳卖点！建议清仓或减至最低仓位")
            score -= 3
        else:
            signals.append("⚡【共振分歧】日周波动放大但方向不一 — 减仓观望，等待方向统一")

    # ---- 规则2: 日线极端波动 ----
    if latest_day_vol > d_mean + 2 * d_std:
        if day_up:
            signals.append("✅【极端买入】日波动超2σ+收阳 — 恐慌后反转，短期强烈看多")
            score += 2
        else:
            signals.append("❌【极端卖出】日波动超2σ+收阴 — 恐慌抛售中，立即减仓避险")
            score -= 2

    # ---- 规则3: 周线异常 ----
    if latest_week_vol > w_mean * 1.5:
        if week_up:
            signals.append("📈【周线启动】周波动放大+收阳 — 中期上涨趋势确立，逢回调加仓")
            score += 1
        else:
            signals.append("📉【周线走弱】周波动放大+收阴 — 中期下跌趋势，反弹即是卖点")
            score -= 1

    # ---- 规则4: 低波动盘整 ----
    if latest_day_vol < d_mean - d_std:
        signals.append("⏸️【低波盘整】日波动低于均值-1σ — 变盘前兆，等待放量突破方向再入场")

    # ---- 规则5: 缩量/低波叠加 ----
    recent_5 = day_vol.tail(5)
    if recent_5.max() < d_mean:
        signals.append("📊【持续低波】近5日波动均低于均值 — 大行情酝酿中，密切跟踪")

    # ---- 综合评级 ----
    if score >= 3:
        grade = "⭐⭐⭐⭐⭐ 强烈买入"
    elif score >= 2:
        grade = "⭐⭐⭐⭐ 建议买入"
    elif score >= 1:
        grade = "⭐⭐⭐ 谨慎看多"
    elif score == 0:
        grade = "⭐⭐ 中性观望"
    elif score >= -1:
        grade = "⭐ 偏弱"
    else:
        grade = "⚠️ 建议回避"

    signals.append(f"\n📋 综合评分: {score:+d} → 综合评级: {grade}")
    return signals


def analyze_stock_volatility(
    code: str,
    name: str = "",
    day_start: str = None,
    week_start: str = None,
) -> dict:
    """
    主入口函数：一键完成数据获取 → 波动计算 → 绘图 → 买卖建议

    参数:
        code: 股票代码，如 "000001"（平安银行）
        name: 股票名称（可选，仅用于图表标题）
        day_start: 日K起始日期，如 "2025-01-01"，默认近60日
        week_start: 周K起始日期，如 "2024-06-01"，默认近52周

    返回:
        {
            "code": "000001",
            "day_data": [...],
            "week_data": [...],
            "day_volatility_summary": {...},
            "week_volatility_summary": {...},
            "signals": [...]
        }
    """
    # 默认日期范围
    if day_start is None:
        day_start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    if week_start is None:
        week_start = (datetime.now() - timedelta(weeks=52)).strftime("%Y-%m-%d")

    # 1. 获取数据（复用 autostock API）
    print(f"📡 正在获取 {code} 日K线数据...")
    day_raw = get_kline_data(code, "day", start_date=day_start)
    print(f"📡 正在获取 {code} 周K线数据...")
    week_raw = get_kline_data(code, "week", start_date=week_start)

    # 2. 计算波动率
    day_data = day_raw.get("data", [])
    week_data = week_raw.get("data", [])

    if not day_data or not week_data:
        return {"error": "未能获取到K线数据，请检查股票代码", "code": code}

    day_df = calc_volatility_df(day_data)
    week_df = calc_volatility_df(week_data)

    # 3. 绘图
    print("📊 正在生成波动率双图...")
    plot_dual_volatility(day_df, week_df, code, name)

    # 4. 生成信号
    signals = generate_signals(day_df, week_df)

    # 5. 构建结果
    result = {
        "code": code,
        "name": name,
        "day_count": len(day_df),
        "week_count": len(week_df),
        "day_volatility_stats": {
            "latest": round(day_df["volatility"].iloc[-1], 2),
            "mean": round(day_df["volatility"].mean(), 2),
            "std": round(day_df["volatility"].std(), 2),
            "max": round(day_df["volatility"].max(), 2),
            "min": round(day_df["volatility"].min(), 2),
        },
        "week_volatility_stats": {
            "latest": round(week_df["volatility"].iloc[-1], 2),
            "mean": round(week_df["volatility"].mean(), 2),
            "max": round(week_df["volatility"].max(), 2),
            "min": round(week_df["volatility"].min(), 2),
        },
        "signals": signals,
    }

    # 打印信号
    print("\n" + "=" * 60)
    print(f"  {code} {name} 波动率分析报告")
    print("=" * 60)
    print(f"  日波动率: 最新={result['day_volatility_stats']['latest']}%, "
          f"均值={result['day_volatility_stats']['mean']}%")
    print(f"  周波动率: 最新={result['week_volatility_stats']['latest']}%, "
          f"均值={result['week_volatility_stats']['mean']}%")
    print("-" * 60)
    print("  📌 买卖信号:")
    for s in signals:
        print(f"    {s}")
    print("=" * 60 + "\n")

    return result


# ============ 使用示例 ============
if __name__ == "__main__":
    # 分析平安银行 (000001) 的波动率
    result = analyze_stock_volatility(
        code="000001",
        name="平安银行",
        day_start="2025-01-01",
        week_start="2024-06-01",
    )
```

# 依赖安装

```bash
pip install requests matplotlib pandas numpy
```

# 快速使用

```python
from stock_volatility import analyze_stock_volatility

# 分析任意A股
result = analyze_stock_volatility("600519", "贵州茅台")
```

# 输出说明

| 输出项 | 说明 |
|--------|------|
| `{code}_volatility.png` | 日波动+周波动双图，红绿柱表示涨跌，含均值线和阈值线 |
| `signals` | 买卖建议列表，按优先级排列 |
| `*_volatility_stats` | 波动率统计（最新值、均值、标准差、极值） |
| 综合评分 | -3 到 +3 的评分，配合 ⭐ 评级给出操作方向 |

# 注意事项

- TOKEN 复用 `autostock` skill 的 `zgaLG8unUPr`
- 日K 默认获取最近 120 个交易日，周K 默认获取最近 52 周
- 前复权数据（`fq_type=1`）为默认，避免除权除息造成的价格跳空
- 波动率均值线 ±2σ 为极端区域，常对应行情反转点
