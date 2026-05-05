# my-agent
基于 FastAPI 和 LangChain 构建的高性能 AI 智能体后端，由 DeepSeek API 驱动。
### 📐 系统逻辑架构 (System Architecture)

该智能体遵循 **ReAct (Reasoning and Acting)** 范式运行：
1. **感知层 (Perception)**：通过 Streamlit 多模态组件获取用户指令及图像输入。
2. **决策层 (Reasoning)**：DeepSeek-V3 接收 System Prompt 指导，自主判断是否需要调用工具。
3. **行动层 (Action)**：
   - **联网工具**：遇到时效性问题自动触发 Tavily Search。
   - **代码工具**：遇到数学或逻辑任务自动触发 Python REPL。
4. **记忆层 (Memory)**：基于 SQLite 实现对话状态持久化，确保跨 Session 的逻辑连贯。
