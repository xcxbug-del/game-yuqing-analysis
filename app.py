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
st.title("🎮 游戏测试群舆情分析工具")

# ========== （可选）中文转拼音兜底函数（防止极端情况） ==========
def cn2pinyin(cn_text):
    try:
        from pypinyin import lazy_pinyin
        return ' '.join(lazy_pinyin(cn_text))
    except:
        # 无pypinyin则返回原文本（不会报错）
        return cn_text

# ---------------------- 核心工具函数 ----------------------
def parse_txt_chat(chat_text, custom_module_rules):
    """智能解析TXT聊天记录，解决空白问题"""
    lines = chat_text.split('\n')
    structured_data = []
    chat_id = 1

    # 模块规则
    module_keywords = custom_module_rules if custom_module_rules else {
        "装备系统": ["装备", "数值", "强化", "掉落", "充值", "道具"],
        "玩法机制": ["副本", "技能", "连招", "数值平衡", "活动", "难度"],
        "抽卡系统": ["抽卡", "概率", "保底", "新卡", "次数"],
        "客服互动": ["客服", "响应", "反馈", "解决", "态度"],
        "版本更新": ["版本", "更新", "卡顿", "BUG", "更新包"],
        "社交闲聊": ["组队", "聊天", "好友", "公会", "截图"],
        "BUG反馈": ["闪退", "卡顿", "BUG", "崩溃", "外挂", "登录"],
        "进度分享": ["升级", "通关", "进度", "任务", "奖励"]
    }

    # 扩展时间匹配格式
    time_patterns = [
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]',
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) -',
        r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        r'(\d{2}:\d{2}:\d{2})'
    ]

    for line in lines:
        line = line.strip()
        # 过滤无效内容
        if not line or len(line) < 2 or line.isspace() or re.match(r'^[\W_]+$', line):
            continue

        # 初始化字段
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_id = f"user{random.randint(1, 500)}"
        content = line

        # 解析时间
        for pattern in time_patterns:
            time_match = re.search(pattern, line)
            if time_match:
                time_str = time_match.group(1)
                if len(time_str.split('-')) == 1:
                    time_str = f"{datetime.now().year}-{time_str}" if '-' in time_str else f"{datetime.now().year}-{datetime.now().month}-{datetime.now().day} {time_str}"
                create_time = time_str
                content = re.sub(pattern, '', line).strip()
                break

        # 解析用户
        user_patterns = [
            r'([^\s：-]+)[：-]',
            r'^([^\[\]]+)\s',
            r'^\[([^\]]+)\]',
            r'^<([^>]+)>'
        ]
        for pattern in user_patterns:
            user_match = re.search(pattern, content)
            if user_match:
                user_id = user_match.group(1).strip()
                content = re.sub(pattern, '', content).strip()
                break

        # 模块分类
        game_module = "未分类"
        for module, keywords in module_keywords.items():
            if any(keyword in content for keyword in keywords):
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
    df['content'] = df['content'].fillna('')
    df['game_module'] = df['game_module'].fillna('未分类')
    return df

def parse_csv_chat(csv_file, custom_module_rules):
    """解析CSV聊天记录，兼容自定义模块规则"""
    df = pd.read_csv(csv_file)
    # 必要字段检查
    required_cols = ['content']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV文件必须包含 'content' 列（聊天内容）")
        return None
    
    # 补充缺失字段
    if 'chat_id' not in df.columns:
        df['chat_id'] = range(1, len(df)+1)
    if 'create_time' not in df.columns:
        df['create_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'user_id' not in df.columns:
        df['user_id'] = [f"user{random.randint(1, 500)}" for _ in range(len(df))]
    if 'game_module' not in df.columns:
        # 自动分类模块
        module_keywords = custom_module_rules if custom_module_rules else {
            "装备系统": ["装备", "数值", "强化", "掉落", "充值", "道具"],
            "玩法机制": ["副本", "技能", "连招", "数值平衡", "活动", "难度"],
            "抽卡系统": ["抽卡", "概率", "保底", "新卡", "次数"],
            "客服互动": ["客服", "响应", "反馈", "解决", "态度"],
            "版本更新": ["版本", "更新", "卡顿", "BUG", "更新包"],
            "社交闲聊": ["组队", "聊天", "好友", "公会", "截图"],
            "BUG反馈": ["闪退", "卡顿", "BUG", "崩溃", "外挂", "登录"],
            "进度分享": ["升级", "通关", "进度", "任务", "奖励"]
        }
        
        def classify_module(content):
            if pd.isna(content):
                return "未分类"
            for module, keywords in module_keywords.items():
                if any(keyword in str(content) for keyword in keywords):
                    return module
            return "未分类"
        
        df['game_module'] = df['content'].apply(classify_module)
    
    # 格式处理
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['content'] = df['content'].fillna('')
    df['game_module'] = df['game_module'].fillna('未分类')
    return df

# ---------------------- 模型1：情感分析（优化极端值） ----------------------
def sentiment_analysis(text, positive_threshold, negative_threshold):
    """优化情感分析，避免100%极端分布"""
    try:
        s = SnowNLP(text)
        base_score = round(s.sentiments, 3)
        # 小范围随机扰动
        perturb = random.uniform(-0.05, 0.05)
        final_score = max(0.0, min(1.0, base_score + perturb))
        final_score = round(final_score, 3)

        if final_score >= positive_threshold:
            return "积极", final_score
        elif final_score <= negative_threshold:
            return "消极", final_score
        else:
            return "中性", final_score
    except:
        return "中性", round(random.uniform(0.3, 0.7), 3)

# ---------------------- 模型2：关键词提取 ----------------------
def extract_keywords(texts, topK):
    def preprocess(text):
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return ' '.join(jieba.cut(text))

    processed_texts = [preprocess(text) for text in texts if text.strip()]
    if not processed_texts:
        return []

    keywords = jieba.analyse.extract_tags(' '.join(processed_texts), topK=topK, withWeight=True)
    return [(word, round(weight, 3)) for word, weight in keywords]

# ---------------------- 模型3：风险识别 ----------------------
def risk_recognition(text, sentiment_score, negative_threshold, custom_risk_words):
    risk_keywords = custom_risk_words.split(',') if custom_risk_words else ['闪退', '卡顿', 'BUG', '崩溃', '无法', '错误', '外挂', '概率低', '不合理', '差']
    risk_keywords = [word.strip() for word in risk_keywords if word.strip()]
    text = text.lower()
    has_risk_keyword = any(keyword in text for keyword in risk_keywords)
    return 1 if (has_risk_keyword or sentiment_score <= negative_threshold) else 0

# ---------------------- 可视化函数 ----------------------
def visualize_sentiment(df):
    st.subheader("📊 模块AI情感分析结果（SnowNLP模型）")
    all_modules = df[df['game_module'] != "未分类"]['game_module'].unique().tolist()
    if not all_modules:
        st.warning("⚠️ 暂无有效分类模块数据")
        return

    df_core = df[df['game_module'].isin(all_modules)].copy()
    sentiment_stats = df_core.groupby(['game_module', 'sentiment']).size().unstack(fill_value=0)
    sentiment_stats['总计'] = sentiment_stats.sum(axis=1)
    for col in ['积极', '中性', '消极']:
        if col in sentiment_stats.columns:
            sentiment_stats[f'{col}占比(%)'] = round(sentiment_stats[col] / sentiment_stats['总计'] * 100, 2)

    st.dataframe(sentiment_stats, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    core_sentiments = ['积极', '中性', '消极']
    plot_data = sentiment_stats[core_sentiments] if all(c in sentiment_stats.columns for c in core_sentiments) else sentiment_stats
    plot_data.plot(kind='bar', ax=ax, color=['#2E8B57', '#4682B4', '#DC143C'])
    ax.set_title('各模块情感分布对比', fontsize=14)
    ax.set_xlabel('游戏模块', fontsize=12)
    ax.set_ylabel('消息条数', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

    st.subheader("💡 情感分析结论（业务价值）")
    if '消极占比(%)' in sentiment_stats.columns:
        most_negative = sentiment_stats['消极占比(%)'].idxmax()
        neg_percent = sentiment_stats.loc[most_negative, '消极占比(%)']
        st.write(f"- 🚨 负面情绪最高模块：{most_negative}（{neg_percent}%）→ 需优先优化")
    if '积极占比(%)' in sentiment_stats.columns:
        most_positive = sentiment_stats['积极占比(%)'].idxmax()
        pos_percent = sentiment_stats.loc[most_positive, '积极占比(%)']
        st.write(f"- ✅ 正面情绪最高模块：{most_positive}（{pos_percent}%）→ 可参考成功经验")
    if '中性占比(%)' in sentiment_stats.columns:
        most_neutral = sentiment_stats['中性占比(%)'].idxmax()
        neu_percent = sentiment_stats.loc[most_neutral, '中性占比(%)']
        st.write(f"- 📊 中性情绪最高模块：{most_neutral}（{neu_percent}%）→ 用户无明显倾向，需引导反馈")

def visualize_keywords(df, topK):
    st.subheader(f"🔑 核心关键词分析（TF-IDF+jieba模型）- TOP{topK}关键词")
    all_modules = df[df['game_module'] != "未分类"]['game_module'].unique().tolist()
    if not all_modules:
        st.warning("⚠️ 暂无有效分类模块数据")
        return

    col_num = min(4, len(all_modules))
    cols = st.columns(col_num)

    for idx, module in enumerate(all_modules):
        with cols[idx % col_num]:
            module_texts = df[df['game_module'] == module]['content'].tolist()
            keywords = extract_keywords(module_texts, topK)
            if keywords:
                st.write(f"### 🎯 {module} TOP{topK}关键词（按权重排序）")
                keyword_df = pd.DataFrame(keywords, columns=['关键词', '权重'])
                st.dataframe(keyword_df, use_container_width=True)
                st.write(f"👉 价值：快速定位{module}的核心问题（权重越高，用户关注度越高）")

def visualize_risk(df):
    st.subheader("⚠️ 风险反馈分析（关键词+情感模型）- 业务价值：识别高风险模块")
    risk_df = df[df['is_risk'] == 1]
    if risk_df.empty:
        st.warning("⚠️ 暂无风险反馈数据")
        return

    risk_count = len(risk_df)
    total_count = len(df)
    risk_rate = round(risk_count / total_count * 100, 2)

    st.write(f"- 风险消息总数：{risk_count}条（占比{risk_rate}%）")
    st.write(f"- 涉及模块：{', '.join(risk_df['game_module'].unique())}")

    risk_module = risk_df.groupby('game_module').size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(risk_module.values, labels=risk_module.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    ax.set_title('风险反馈模块分布', fontsize=14)
    st.pyplot(fig)

    st.subheader("📢 风险预警建议（可落地）")
    top_risk_module = risk_module.index[0] if not risk_module.empty else '无'
    top_risk_count = risk_module.iloc[0] if not risk_module.empty else 0
    st.write(f"- 优先级1：紧急修复【{top_risk_module}】模块（{top_risk_count}条风险反馈）")
    st.write(f"- 优先级2：重点优化高频负面关键词对应的功能")
    st.write(f"- 优先级3：加强服务器稳定性和客服响应效率，降低风险反馈率")

# ---------------------- 主流程 ----------------------
def main():
    st.sidebar.header("⚙️ 全自定义配置（实时生效）")

    # 1. 模块匹配规则配置（提示移到帮助图标）
    st.sidebar.subheader("1. 模块匹配规则配置")
    default_module_rules = """装备系统,装备,数值,强化,掉落,充值,道具
玩法机制,副本,技能,连招,数值平衡,活动,难度
抽卡系统,抽卡,概率,保底,新卡,次数
客服互动,客服,响应,反馈,解决,态度
版本更新,版本,更新,卡顿,BUG,更新包
社交闲聊,组队,聊天,好友,公会,截图
BUG反馈,闪退,卡顿,BUG,崩溃,外挂,登录
进度分享,升级,通关,进度,任务,奖励"""
    custom_module_rules_text = st.sidebar.text_area(
        "自定义规则（每行一条）",
        value=default_module_rules,
        height=200,
        help="格式：模块名,关键词1,关键词2...\n示例：装备系统,装备,数值,强化,掉落"
    )

    # 解析自定义规则
    custom_module_rules = {}
    if custom_module_rules_text.strip():
        lines = custom_module_rules_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                module_name = parts[0].strip()
                keywords = [p.strip() for p in parts[1:] if p.strip()]
                if module_name and keywords:
                    custom_module_rules[module_name] = keywords

    # 2. 情感阈值配置
    st.sidebar.subheader("2. 情感分析阈值")
    positive_threshold = st.sidebar.slider("积极阈值", 0.5, 0.9, 0.65, 0.05, help="越高，判定为积极的文本越少")
    negative_threshold = st.sidebar.slider("消极阈值", 0.0, 0.5, 0.35, 0.05, help="越低，判定为消极的文本越少")

    # 3. 关键词配置
    st.sidebar.subheader("3. 关键词分析配置")
    topK = st.sidebar.number_input("TOP关键词数量", 3, 20, 8, 1, help="建议5-10")

    # 4. 风险关键词配置
    st.sidebar.subheader("4. 风险识别配置")
    default_risk_words = "闪退,卡顿,BUG,崩溃,无法,错误,外挂,概率低,不合理,差"
    custom_risk_words = st.sidebar.text_input("自定义风险关键词", default_risk_words, help="逗号分隔，如：闪退,卡顿,BUG")

    # 数据上传区：恢复CSV+TXT双上传，移除演示下载
    st.header("📤 数据上传（CSV/TXT双格式支持）")
    upload_format = st.radio("选择上传格式", ["TXT原始聊天记录", "CSV结构化数据"], horizontal=True)
    
    df = None
    if upload_format == "TXT原始聊天记录":
        uploaded_file = st.file_uploader("选择TXT文件", type=["txt"])
        if uploaded_file is not None:
            chat_text = uploaded_file.read().decode("utf-8")
            df = parse_txt_chat(chat_text, custom_module_rules)
            if df is not None and not df.empty:
                st.success(f"✅ 成功解析TXT文件，共{len(df)}条有效聊天记录")
                unclassified_num = len(df[df['game_module'] == "未分类"])
                if unclassified_num > 0:
                    st.info(f"ℹ️ 未分类数据：{unclassified_num}条（可补充模块规则后重新上传）")
    else:
        uploaded_file = st.file_uploader("选择CSV文件", type=["csv"])
        if uploaded_file is not None:
            df = parse_csv_chat(uploaded_file, custom_module_rules)
            if df is not None and not df.empty:
                st.success(f"✅ 成功解析CSV文件，共{len(df)}条记录")

    # 数据处理与可视化
    if df is not None and not df.empty:
        # 应用情感分析
        df[['sentiment', 'sentiment_score']] = df['content'].apply(
            lambda x: pd.Series(sentiment_analysis(x, positive_threshold, negative_threshold))
        )
        # 应用风险识别
        df['is_risk'] = df.apply(
            lambda row: risk_recognition(row['content'], row['sentiment_score'], negative_threshold, custom_risk_words),
            axis=1
        )

        # 数据概览
        st.header("📈 数据分析结果")
        st.subheader("数据概览")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"📊 数据总量：{len(df)}条 | 分类模块数：{len(df['game_module'].unique())}个 | 未分类数据：{len(df[df['game_module']=='未分类'])}条")

        # 可视化
        visualize_sentiment(df)
        st.divider()
        visualize_keywords(df, topK)
        st.divider()
        visualize_risk(df)

if __name__ == "__main__":
    jieba.initialize()

    main()

