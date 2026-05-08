import streamlit as st
import httpx
import base64

st.set_page_config(page_title="GLM Vision Agent", layout="wide")
st.title("👁️ 智谱真视觉 (多轮版)")

with st.sidebar:
    st.header("多模态输入")
    uploaded_file = st.file_uploader("上传截图或照片...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="已上传预览")
    if st.button("清除对话记录"):
        st.session_state.chat_history = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 展示历史对话
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("问问 Agent..."):
    # 1. 记录并显示用户输入
    st.chat_message("user").markdown(prompt)

    img_b64 = None
    if uploaded_file:
        img_b64 = base64.b64encode(uploaded_file.getvalue()).decode()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            # 2. 发送包含 history 的 payload
            payload = {
                "prompt": prompt, 
                "image_data": img_b64,
                "history": st.session_state.chat_history 
            }
            with httpx.stream("POST", "http://localhost:8000/chat", json=payload, timeout=60.0) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        content = line[6:]
                        full_response += content
                        placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            # 3. 把这一轮对话存入历史记录
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"连接失败: {e}")
