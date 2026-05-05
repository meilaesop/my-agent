import os
import streamlit as st
import pytz
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="视觉全能 Agent", page_icon="👁️", layout="wide")
st.title("👁️ 视觉 + 联网 + 内存 Agent")

# 1. 基础配置
tz = pytz.timezone('Asia/Shanghai')
now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base='https://api.deepseek.com'
)

search_tool = TavilySearchResults(k=3)
python_tool = PythonREPLTool()
llm_with_tools = llm.bind_tools([search_tool, python_tool])

# 2. 侧边栏：处理图片输入
with st.sidebar:
    st.header("多模态输入")
    uploaded_file = st.file_uploader("上传一张图片...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Agent 正在观察这张图", width=200)

# 3. 对话逻辑
if "msgs" not in st.session_state:
    st.session_state.msgs = [SystemMessage(content=f"你是全能助手，当前时间 {now_str}。你可以看到用户上传的图片预览。")]

for m in st.session_state.msgs:
    if isinstance(m, (HumanMessage, AIMessage)):
        st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").write(m.content)

if prompt := st.chat_input("描述图片内容或提出复杂问题..."):
    st.chat_message("user").write(prompt)
    
    # 如果有图，给 Prompt 加点料
    full_prompt = prompt
    if uploaded_file:
        full_prompt = f"【用户上传了图片: {uploaded_file.name}】\n问题：{prompt}"
    
    st.session_state.msgs.append(HumanMessage(content=full_prompt))

    with st.chat_message("assistant"):
        with st.status("🧠 跨模态思考中...") as s:
            res = llm_with_tools.invoke(st.session_state.msgs)
            
            while res.tool_calls:
                st.session_state.msgs.append(res)
                for tool in res.tool_calls:
                    if tool["name"] == "tavily_search_results_json":
                        st.write("🔍 正在搜索背景资料...")
                        out = search_tool.invoke(tool["args"]["query"])
                    else:
                        st.write("💻 正在运行计算逻辑...")
                        out = python_tool.invoke(tool["args"]["query"])
                    st.session_state.msgs.append(ToolMessage(tool_call_id=tool["id"], content=str(out)))
                res = llm_with_tools.invoke(st.session_state.msgs)
            
            s.update(label="处理完毕!", state="complete")
            st.write(res.content)
            st.session_state.msgs.append(AIMessage(content=res.content))
