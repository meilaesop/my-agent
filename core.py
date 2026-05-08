import os
import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# 加载 .env 中的 API 密钥
load_dotenv()

# --- 初始化工具组件 ---
# 信号量控制：限制同时进行的 API 请求数量（防止触发频率限制）
sem = asyncio.Semaphore(5)
# 搜索工具：用于联网获取最新信息
search_tool = TavilySearchResults(k=3)
# 代码执行器：可以在本地环境运行 Python 代码并返回输出结果
python_executor = PythonREPL()

async def get_agent_response(prompt: str, image_base64: str = None, history: list = None):
    """
    Agent 核心调度函数：实现 思考(Reasoning) -> 行动(Action) 逻辑
    """
    
    # 1. 初始化模型：DeepSeek 负责逻辑推理，GLM 负责视觉理解
    reasoning_llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com'
    )
    
    vision_llm = ChatOpenAI(
        model='glm-4v', 
        openai_api_key=os.getenv("ZHIPUAI_API_KEY"), 
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/'
    )

    yield "data: 【🧠 Agent 正在规划任务路径...】\n\n"

    # --- 阶段 1：决策路由 (Decision Layer) ---
    # 让 LLM 判断当前问题需要调用哪些工具，避免浪费 API 额度
    decision_prompt = (
        f"用户问题：{prompt}\n"
        "请作为调度员判断是否需要以下工具：\n"
        "1. search: 涉及实时新闻、天气或未知事实\n"
        "2. vision: 用户上传了图片且问题与图片相关\n"
        "3. code: 涉及数学计算、复杂逻辑或数据处理\n"
        "请仅回复 JSON 格式：{\"search\": \"yes/no\", \"vision\": \"yes/no\", \"code\": \"yes/no\"}"
    )
    
    try:
        decision_res = await reasoning_llm.ainvoke([HumanMessage(content=decision_prompt)])
        # 尝试解析模型输出的 JSON 决策
        decision = json.loads(decision_res.content)
    except:
        # 解析失败的兜底策略：如果上传了图片就看图，否则默认只尝试搜索
        decision = {"search": "yes", "vision": "yes" if image_base64 else "no", "code": "no"}

    # --- 阶段 2：异步动作执行 (Action Layer) ---
    # 使用 asyncio.gather 并行执行所有选中的任务，大幅缩短等待时间

    async def task_vision():
        """视觉分析任务"""
        if decision.get("vision") == "yes" and image_base64:
            async with sem:
                content = [
                    {"type": "text", "text": "请描述图中的关键细节，辅助回答用户问题。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                res = await vision_llm.ainvoke([HumanMessage(content=content)])
                return res.content
        return "未触发视觉任务"

    async def task_search():
        """联网搜索任务"""
        if decision.get("search") == "yes":
            async with sem:
                try:
                    # 设置 10 秒超时防止网络卡死
                    return await asyncio.wait_for(asyncio.to_thread(search_tool.invoke, prompt), timeout=10.0)
                except:
                    return "搜索服务暂时不可用"
        return "未触发搜索任务"

    async def task_code():
        """代码执行任务 (Python REPL)"""
        if decision.get("code") == "yes":
            # 先让模型写出 Python 代码
            code_gen_res = await reasoning_llm.ainvoke([
                HumanMessage(content=f"针对问题 '{prompt}'，编写 Python 代码。只需输出代码，不要任何文字解释。")
            ])
            code = code_gen_res.content.strip().replace("```python", "").replace("```", "")
            
            # 在本地 REPL 环境中执行代码并捕获 print 输出
            try:
                # 使用 loop.run_in_executor 防止同步的 REPL 阻塞异步主循环
                loop = asyncio.get_event_loop()
                exec_result = await loop.run_in_executor(None, python_executor.run, code)
                return f"代码运行结果：\n{exec_result}"
            except Exception as e:
                return f"代码执行出错：{str(e)}"
        return "未触发代码任务"

    # 并发启动三个任务，等待它们全部返回
    vision_info, search_info, code_info = await asyncio.gather(
        task_vision(), task_search(), task_code()
    )

    # --- 阶段 3：记忆管理 (Memory) ---
    # 仅提取最近 6 条历史，保持上下文专注并节省 Token
    recent_history = history[-6:] if history else []
    messages = [SystemMessage(content="你是一个具备视觉、搜索和代码能力的智能 Agent。")]
    
    for m in recent_history:
        if m["role"] == "user": messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant": messages.append(AIMessage(content=m["content"]))

    # --- 阶段 4：最终结果整合 (Final Generation) ---
    final_prompt = f"""
    参考信息：
    - 视觉背景：{vision_info}
    - 实时搜索：{search_info}
    - 代码计算：{code_info}
    
    当前用户问题：{prompt}
    请综合上述参考信息，给出最终的专业回答。
    """
    messages.append(HumanMessage(content=final_prompt))
    
    # 开启流式模型进行最终回复
    streaming_llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com',
        streaming=True
    )
    
    async for chunk in streaming_llm.astream(messages):
        if chunk.content:
            yield f"data: {chunk.content}\n\n"
