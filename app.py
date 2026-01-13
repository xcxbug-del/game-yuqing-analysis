import streamlit as st
import pandas as pd
import jieba
import jieba.analyse
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
import re
from datetime import datetime
import matplotlib.font_manager as fm
import os

# ========== 基础配置 ==========
warnings.filterwarnings('ignore')

# ========== Ubuntu云端中文显示核心配置（无报错版） ==========
def setup_chinese_font():
    # 优先使用Ubuntu预装中文字体
    font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    for font in font_list:
        try:
            # 全局设置字体
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            # 验证字体是否生效
            fig, ax = plt.subplots(figsize=(1,1))
            ax.text(0.5, 0.5, '测试中文')
            plt.close(fig)
            print(f"✅ 字体加载成功：{font}")
            return
        except:
            continue
    print("⚠️ 无中文字体，中文可能显示方框")

setup_chinese_font()

# ========== 页面配置 ==========
st.set_page_config(page_title="游戏测试群舆情分析工具", layout="wide")
st.title("🎮 游戏测试群舆情分析工具")

# ========== 核心函数 ==========
def parse_txt_chat(chat_text):
    lines = chat_text.split('\n')
    structured_data = []
    chat_id = 1
    module_keywords = {
        "装备系统": ["装备", "数值", "强化", "掉落", "充值", "道具"],
        "玩法机制": ["副本", "技能", "连招", "数值平衡", "活动", "难度"],
        "抽卡系统": ["抽卡", "概率", "保底", "新卡", "次数"],
        "客服互动": ["客服", "响应", "反馈", "解决", "态度"],
        "版本更新": ["版本", "更新", "卡顿", "BUG", "更新包"],
        "社交闲聊": ["组队", "聊天", "好友", "公会", "截图"],
        "BUG反馈": ["闪退", "卡顿", "BUG", "崩溃", "外挂", "登录"],
        "进度分享": ["升级", "通关", "进度", "任务", "奖励"]
    }
    time_patterns = [r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', r'(\d{2}:\d{2}:\d{2})']
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_id = f"user{random.randint(1, 500)}"
        content = line
        
        # 提取时间
        for pattern in time_patterns:
            match = re.search(pattern, line)
            if match:
                create_time = match.group(1)
                content = re.sub(pattern, '', line).strip()
                break
        
        # 提取用户
        user_patterns = [r'([^\s：-]+)[：-]', r'^\[([^\]]+)\]']
        for pattern in user_patterns:
            match = re.search(pattern, content)
            if match:
                user_id = match.group(1).strip()
                content = re.sub(pattern, '', content).strip()
                break
        
        # 分类模块
        game_module = "未分类"
        for module, keywords in module_keywords.items():
            if any(k in content for k in keywords):
                game_module = module
                break
        
        structured_data.append({
            "chat_id": chat_id,
            "create_time": create_time,
            "user_id": user_id,
            "content": content,
            "game_module": game_module
        })
        chat_id += 1
    
    df = pd.DataFrame(structured_data)
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    return df

def sentiment_analysis(text):
    try:
        s = SnowNLP(text)
        score = round(s.sentiments, 3)
        if score >= 0.65:
            return "积极", score
        elif score <= 0.35:
            return "消极", score
        else:
            return "中性", score
    except:
        return "中性", 0.5

def visualize_sentiment(df):
    st.subheader("📊 模块情感分析")
    modules = ["装备系统", "玩法机制", "抽卡系统", "客服互动", "版本更新", "社交闲聊", "BUG反馈", "进度分享"]
    df_core = df[df['game_module'].isin(modules)]
    
    # 统计情感
    sentiment_stats = df_core.groupby(['game_module', 'sentiment']).size().unstack(fill_value=0)
    sentiment_stats = sentiment_stats.reindex(modules, fill_value=0)
    
    # 绘图（纯中文，无额外字体参数）
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiment_stats[['积极', '中性', '消极']].plot(kind='bar', ax=ax, color=['#2E8B57', '#4682B4', '#DC143C'])
    ax.set_title('各模块情感分布', fontsize=14)
    ax.set_xlabel('游戏模块', fontsize=12)
    ax.set_ylabel('消息数量', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

# ========== 主流程 ==========
def main():
    # 上传文件
    uploaded_file = st.file_uploader("上传TXT聊天记录", type=["txt"])
    if uploaded_file:
        chat_text = uploaded_file.read().decode("utf-8")
        df = parse_txt_chat(chat_text)
        
        # 情感分析
        df[['sentiment', 'score']] = df['content'].apply(lambda x: pd.Series(sentiment_analysis(x)))
        
        # 显示结果
        st.success(f"✅ 解析成功，共{len(df)}条记录")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 可视化
        visualize_sentiment(df)

if __name__ == "__main__":
    jieba.initialize()
    main()
