# 🤖 My-Agent: 多模态 AI 智能体开发项目

这是一个专注于 AI Agent 开发的实战项目，基于 **FastAPI** 和 **LangChain** 构建，具备视觉、搜索、计算三合一能力的超级助手。

## 🌟 核心能力
- **👁️ 多模态视觉 (GLM-4V)**：精准识别代码截图、UI 界面及复杂场景。
- **🌐 实时搜索 (Tavily)**：实时获取互联网最新信息，弥补模型知识滞后。
- **🐍 自动化代码执行 (REPL)**：Agent 自主编写并运行 Python 代码解决逻辑计算任务。
- **💾 异步持久化 (SQLAlchemy)**：集成 SQLite/MySQL 异步驱动，完整记录对话上下文。

## 🛠️ 技术栈清单 (Stack)
- **核心框架**: FastAPI (异步高性能 Web 框架)
- **Agent 编排**: LangChain / LangGraph (ReAct 架构)
- **大模型**: DeepSeek-V3 (核心推理), GLM-4V (多模态视觉)
- **异步处理**: Asyncio, Python-dotenv
- **数据库**: SQLAlchemy (Async), SQLite (本地持久化)
- **开发工具链**: GitHub Codespaces, Git, Linux (Termux/Android 适配)
- **前端交互**: 基于 SSE (Server-Sent Events) 的流式打字机效果

## 🚀 开发者指南
1. **环境准备**: `pip install -r requirements.txt`
2. **密钥配置**: 在 `.env` 中配置 `DEEPSEEK_API_KEY`, `ZHIPUAI_API_KEY`, `TAVILY_API_KEY`
3. **运行启动**: `python app.py`

## 📝 学习历程
- [x] Django 项目起步与模型迁移
- [x] DRF (Django REST Framework) 权限与认证
- [x] 迁移至异步 FastAPI 架构
- [x] 集成多模态 Vision 能力与 REPL 工具
