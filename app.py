# ---------------------- 全局配置（无报错+Ubuntu云端中文显示） ----------------------
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

warnings.filterwarnings('ignore')

# ========== 核心：适配所有matplotlib版本的Ubuntu中文字体配置（无报错） ==========
def setup_ubuntu_chinese_font():
    # 方案1：直接指定Ubuntu预装中文字体名称（无需路径/缓存，最稳定）
    chinese_font_names = [
        'WenQuanYi Micro Hei',  # Ubuntu预装核心中文字体
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',     # 新版Ubuntu预装
        'DejaVu Sans'           # 兜底英文字体
    ]
    
    # 遍历字体列表，找到第一个可用的中文字体
    for font_name in chinese_font_names:
        try:
            # 测试字体是否可用（绘制隐藏文本验证）
            fig, ax = plt.subplots(figsize=(1,1))
            ax.text(0.5, 0.5, '测试中文显示', fontname=font_name)
            plt.close(fig)
            
            # 全局设置（所有绘图默认用该字体）
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
            print(f"✅ 成功加载中文字体：{font_name}")
            return
        except Exception as e:
            print(f"⚠️ 字体 {font_name} 不可用：{str(e)}")
            continue
    
    # 终极兜底：即使无中文字体，也不报错（显示原文本）
    print("⚠️ 无可用中文字体，将使用默认字体（中文可能显示方框）")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 立即执行字体配置（必须在所有绘图代码前）
setup_ubuntu_chinese_font()

# ---------------------- 原有配置保留 ----------------------
st.set_page_config(page_title="游戏测试群舆情分析工具", layout="wide")
st.title("🎮 游戏测试群舆情分析工具（面试版）")
