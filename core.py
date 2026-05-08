import os
import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# 1. 资源限制：控制并发，防止被 API 封锁
sem = asyncio.Semaphore(5)
search_tool = TavilySearchResults(k=3)

async def get_agent_response(prompt: str, image_base64: str = None, history: list = None):
    # 逻辑模型 (DeepSeek) - 负责决策和最终总结
    reasoning_llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com'
    )
    
    # 视觉模型 (智谱)
    vision_llm = ChatOpenAI(
        model='glm-4v', 
        openai_api_key=os.getenv("ZHIPUAI_API_KEY"), 
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/'
    )

    yield "data: 【🧠 正在思考处理方案...】\n\n"

    # --- 阶段 1：决策路由 (Decision Layer) ---
    # 这里的技巧是：先让模型判断是否需要图片分析和搜索
    decision_prompt = f"用户问题：{prompt}\n请判断：1.是否需要联网搜索(yes/no) 2.是否需要分析上传的图片(yes/no)。仅回复JSON，如 {{\"search\": \"yes\", \"vision\": \"yes\"}}"
    try:
        decision_res = await reasoning_llm.ainvoke([HumanMessage(content=decision_prompt)])
        decision = json.loads(decision_res.content)
    except:
        decision = {"search": "yes", "vision": "yes" if image_base64 else "no"}

    # --- 阶段 2：异步并发执行 (Execution Layer) ---
    tasks = []
    
    async def task_vision():
        if decision.get("vision") == "yes" and image_base64:
            async with sem: # 并发控制
                content = [
                    {"type": "text", "text": "描述图中对解决问题有帮助的细节。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                res = await vision_llm.ainvoke([HumanMessage(content=content)])
                return res.content
        return "无图片背景"

    async def task_search():
        if decision.get("search") == "yes":
            async with sem:
                try:
                    # 设置 10 秒超时
                    return await asyncio.wait_for(asyncio.to_thread(search_tool.invoke, prompt), timeout=10.0)
                except asyncio.TimeoutError:
                    return "搜索超时，将基于已有知识回答。"
        return "无需搜索"

    # 同时启动
    vision_desc, search_results = await asyncio.gather(task_vision(), task_search())

    # --- 阶段 3：多轮对话裁剪 (Memory Management) ---
    # 仅保留最近 6 轮对话，防止 Token 爆炸
    recent_history = history[-6:] if history else []
    messages = [SystemMessage(content="你是一个具备自主决策能力的 Agent。")]
    
    for msg in recent_history:
        if msg["role"] == "user": messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": messages.append(AIMessage(content=msg["content"]))

    # --- 阶段 4：最终生成 (Generation) ---
    final_input = f"""
    【视觉观察】：{vision_desc}
    【实时搜索】：{search_results}
    【当前问题】：{prompt}
    请整合以上信息。如果涉及代码建议，请按“修复建议、代码实现、注意事项”三个部分回答。
    """
    messages.append(HumanMessage(content=final_input))
    
    # 使用流式输出
    streaming_llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com',
        streaming=True
    )
    
    async for chunk in streaming_llm.astream(messages):
        if chunk.content:
            yield f"data: {chunk.content}\n\n"
