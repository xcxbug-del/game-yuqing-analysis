import streamlit as st
import pandas as pd
import jieba
import jieba.analyse
from snownlp import SnowNLP
import warnings
import random
import re
from datetime import datetime
import numpy as np

# ========== 基础配置 ==========
warnings.filterwarnings('ignore')
st.set_page_config(page_title="游戏测试群舆情分析工具", layout="wide")
st.title("🎮 游戏测试群舆情分析工具（面试完整版）")

# ========== 工具函数：安全处理数值 ==========
def safe_divide(a, b, default=0.0):
    """安全除法，避免除零错误"""
    try:
        return a / b if b != 0 else default
    except:
        return default

def clamp_value(value, min_val=0.0, max_val=1.0):
    """将数值限制在指定范围"""
    return max(min_val, min(max_val, value))

# ========== 核心解析函数（完全保留）==========
def parse_txt_chat(chat_text, custom_module_rules):
    lines = chat_text.split('\n')
    structured_data = []
    chat_id = 1
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
    time_patterns = [
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]',
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) -',
        r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        r'(\d{2}:\d{2}:\d{2})'
    ]
    for line in lines:
        line = line.strip()
        if not line or len(line) < 2 or line.isspace() or re.match(r'^[\W_]+$', line):
            continue
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_id = f"user{random.randint(1, 500)}"
        content = line
        for pattern in time_patterns:
            time_match = re.search(pattern, line)
            if time_match:
                time_str = time_match.group(1)
                if len(time_str.split('-')) == 1:
                    time_str = f"{datetime.now().year}-{time_str}" if '-' in time_str else f"{datetime.now().year}-{datetime.now().month}-{datetime.now().day} {time_str}"
                create_time = time_str
                content = re.sub(pattern, '', line).strip()
                break
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
    try:
        df = pd.read_csv(csv_file)
        required_cols = ['content']
        if not all(col in df.columns for col in required_cols):
            st.error("❌ CSV文件必须包含 'content' 列（聊天内容）")
            return None
        if 'chat_id' not in df.columns:
            df['chat_id'] = range(1, len(df)+1)
        if 'create_time' not in df.columns:
            df['create_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 'user_id' not in df.columns:
            df['user_id'] = [f"user{random.randint(1, 500)}" for _ in range(len(df))]
        if 'game_module' not in df.columns:
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
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        df['content'] = df['content'].fillna('')
        df['game_module'] = df['game_module'].fillna('未分类')
        return df
    except Exception as e:
        st.error(f"❌ CSV解析失败：{str(e)}")
        return None

def sentiment_analysis(text, positive_threshold, negative_threshold):
    try:
        s = SnowNLP(text)
        base_score = round(s.sentiments, 3)
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

def extract_keywords(texts, topK):
    def preprocess(text):
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return ' '.join(jieba.cut(text))
    processed_texts = [preprocess(text) for text in texts if text.strip()]
    if not processed_texts:
        return []
    keywords = jieba.analyse.extract_tags(' '.join(processed_texts), topK=topK, withWeight=True)
    return [(word, round(weight, 3)) for word, weight in keywords]

def risk_recognition(text, sentiment_score, negative_threshold, custom_risk_words):
    risk_keywords = custom_risk_words.split(',') if custom_risk_words else ['闪退', '卡顿', 'BUG', '崩溃', '无法', '错误', '外挂', '概率低', '不合理', '差']
    risk_keywords = [word.strip() for word in risk_keywords if word.strip()]
    text = text.lower()
    has_risk_keyword = any(keyword in text for keyword in risk_keywords)
    return 1 if (has_risk_keyword or sentiment_score <= negative_threshold) else 0

# ========== 可视化优化：美观+无报错 ==========
def show_sentiment_analysis(df):
    st.subheader("📊 模块AI情感分析结果（SnowNLP模型）")
    all_modules = df[df['game_module'] != "未分类"]['game_module'].unique().tolist()
    DEFAULT_8_MODULES = ["装备系统", "玩法机制", "抽卡系统", "客服互动", "版本更新", "社交闲聊", "BUG反馈", "进度分享"]
    if not all_modules:
        st.warning("⚠️ 暂无有效分类模块数据")
        return
    
    df_core = df[df['game_module'].isin(DEFAULT_8_MODULES)].copy()
    # ========== 关键修复：强制指定情感分类，确保三列都存在 ==========
    sentiment_categories = ['积极', '中性', '消极']
    df_core['sentiment'] = pd.Categorical(df_core['sentiment'], categories=sentiment_categories)
    
    # 统计情感数据
    sentiment_stats = df_core.groupby(['game_module', 'sentiment']).size().unstack(fill_value=0)
    # 确保列顺序正确
    sentiment_stats = sentiment_stats.reindex(columns=sentiment_categories, fill_value=0)
    
    sentiment_stats['总计'] = sentiment_stats.sum(axis=1)
    
    # 安全计算占比
    for col in sentiment_categories:
        if col in sentiment_stats.columns:
            sentiment_stats[f'{col}占比(%)'] = round(safe_divide(sentiment_stats[col], sentiment_stats['总计']) * 100, 2)
    
    # ========== 新增调试输出：查看原始情感分布 ==========
    st.subheader("🔍 调试：整体情感分布")
    overall_sentiment = df_core['sentiment'].value_counts()
    st.dataframe(overall_sentiment, use_container_width=True)
    
    # 2. 优化版消极占比展示（卡片式+安全进度条）
    st.subheader("⚠️ 各模块消极占比（重点关注）")
    col_num = 4
    cols = st.columns(col_num)
    module_list = [m for m in DEFAULT_8_MODULES if m in sentiment_stats.index]
    
    for idx, module in enumerate(module_list):
        with cols[idx % col_num]:
            # 安全获取消极占比
            neg_rate = sentiment_stats.loc[module, '消极占比(%)'] if '消极占比(%)' in sentiment_stats.columns else 0.0
            # 安全进度条值（限制0-1）
            progress_val = clamp_value(safe_divide(neg_rate, 100))
            
            # 卡片式展示（美观）
            with st.container(border=True):
                st.markdown(f"### 🎮 {module}")
                st.progress(progress_val, text=f"消极占比：{neg_rate}%")
                
                # 风险等级标签
                if neg_rate > 30:
                    st.markdown(f"<span style='color:red; font-weight:bold;'>🔴 高风险</span>", unsafe_allow_html=True)
                elif neg_rate > 15:
                    st.markdown(f"<span style='color:orange; font-weight:bold;'>🟡 中风险</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:green; font-weight:bold;'>🟢 低风险</span>", unsafe_allow_html=True)
    
    # 3. 分析结论
    st.subheader("💡 情感分析结论（业务价值）")
    if '消极占比(%)' in sentiment_stats.columns:
        most_negative = sentiment_stats['消极占比(%)'].idxmax()
        neg_percent = sentiment_stats.loc[most_negative, '消极占比(%)']
        st.error(f"🚨 负面情绪最高模块：{most_negative}（{neg_percent}%）→ 需优先优化")
    if '积极占比(%)' in sentiment_stats.columns:
        most_positive = sentiment_stats['积极占比(%)'].idxmax()
        pos_percent = sentiment_stats.loc[most_positive, '积极占比(%)']
        st.success(f"✅ 正面情绪最高模块：{most_positive}（{pos_percent}%）→ 可参考成功经验")

def show_keywords_analysis(df, topK):
    st.subheader(f"🔑 核心关键词分析（TF-IDF+jieba模型）- TOP{topK}关键词")
    all_modules = df[df['game_module'] != "未分类"]['game_module'].unique().tolist()
    DEFAULT_8_MODULES = ["装备系统", "玩法机制", "抽卡系统", "客服互动", "版本更新", "社交闲聊", "BUG反馈", "进度分享"]
    all_modules = [m for m in DEFAULT_8_MODULES if m in all_modules]
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
                # 卡片式关键词展示
                with st.container(border=True):
                    st.markdown(f"### 🎯 {module}")
                    # 关键词表格化展示
                    keyword_df = pd.DataFrame(keywords, columns=['关键词', '权重'])
                    st.dataframe(keyword_df, use_container_width=True, hide_index=True)

def show_risk_analysis(df):
    st.subheader("⚠️ 风险反馈分析（关键词+情感模型）- 业务价值：识别高风险模块")
    risk_df = df[df['is_risk'] == 1]
    if risk_df.empty:
        st.success("✅ 暂无风险反馈数据")
        return
    
    risk_count = len(risk_df)
    total_count = len(df)
    risk_rate = round(safe_divide(risk_count, total_count) * 100, 2)
    
    # 核心风险数据（美化）
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="风险消息总数", value=f"{risk_count}条", delta=f"占比{risk_rate}%", delta_color="inverse")
    with col2:
        st.metric(label="涉及模块数", value=f"{len(risk_df['game_module'].unique())}个")
    
    # 风险模块排名（美化成表格）
    st.subheader("📊 风险模块排名")
    risk_module = risk_df.groupby('game_module').size().sort_values(ascending=False).reset_index(name='风险条数')
    # 添加风险等级
    risk_module['风险等级'] = risk_module['风险条数'].apply(lambda x: '高风险' if x >= 5 else '中风险' if x >= 2 else '低风险')
    # 表格美化
    st.dataframe(
        risk_module,
        use_container_width=True,
        column_config={
            "game_module": st.column_config.TextColumn("模块名称", width="medium"),
            "风险条数": st.column_config.NumberColumn("风险条数", format="%d条"),
            "风险等级": st.column_config.SelectboxColumn("风险等级", options=["高风险", "中风险", "低风险"])
        }
    )
    
    # 风险建议（美化）
    st.subheader("📢 风险预警建议（可落地）")
    top_risk_module = risk_module.iloc[0]['game_module'] if not risk_module.empty else '无'
    top_risk_count = risk_module.iloc[0]['风险条数'] if not risk_module.empty else 0
    # 卡片式建议
    with st.container(border=True):
        st.markdown(f"""
        <div style='line-height: 1.8;'>
        <p>🔴 <strong>优先级1</strong>：紧急修复【{top_risk_module}】模块（{top_risk_count}条风险反馈）</p>
        <p>🟡 <strong>优先级2</strong>：重点优化高频负面关键词对应的功能</p>
        <p>🟢 <strong>优先级3</strong>：加强服务器稳定性和客服响应效率，降低风险反馈率</p>
        </div>
        """, unsafe_allow_html=True)

# ========== 主流程（完全保留）==========
def main():
    st.sidebar.header("⚙️ 全自定义配置（实时生效）")
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
    st.sidebar.subheader("2. 情感分析阈值")
    positive_threshold = st.sidebar.slider("积极阈值", 0.5, 0.9, 0.65, 0.05, help="越高，判定为积极的文本越少")
    negative_threshold = st.sidebar.slider("消极阈值", 0.9, 0.5, 0.35, 0.05, help="越低，判定为消极的文本越少")
    st.sidebar.subheader("3. 关键词分析配置")
    topK = st.sidebar.number_input("TOP关键词数量", 3, 20, 8, 1, help="建议5-10")
    st.sidebar.subheader("4. 风险识别配置")
    default_risk_words = "闪退,卡顿,BUG,崩溃,无法,错误,外挂,概率低,不合理,差"
    custom_risk_words = st.sidebar.text_input("自定义风险关键词", default_risk_words, help="逗号分隔，如：闪退,卡顿,BUG")
    
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
    
    if df is not None and not df.empty:
        df[['sentiment', 'sentiment_score']] = df['content'].apply(
            lambda x: pd.Series(sentiment_analysis(x, positive_threshold, negative_threshold))
        )
        df['is_risk'] = df.apply(
            lambda row: risk_recognition(row['content'], row['sentiment_score'], negative_threshold, custom_risk_words),
            axis=1
        )
        st.header("📈 数据分析结果")
        st.subheader("数据概览")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"📊 数据总量：{len(df)}条 | 分类模块数：{len(df['game_module'].unique())}个 | 未分类数据：{len(df[df['game_module']=='未分类'])}条")
        
        show_sentiment_analysis(df)
        st.divider()
        show_keywords_analysis(df, topK)
        st.divider()
        show_risk_analysis(df)

if __name__ == "__main__":
    jieba.initialize()
    main()

