import os
import time
from typing import Optional, List
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text
from core import get_agent_response

# --- 1. 工业级异步数据库配置 ---
# 使用 aiosqlite 驱动，确保数据库操作不阻塞 AI 流式输出
DATABASE_URL = "sqlite+aiosqlite:///./agent_storage.db"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# --- 2. 数据库模型定义 (ORM) ---
class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String(50))
    prompt = Column(Text)
    response_summary = Column(Text)

app = FastAPI()

# --- 3. 自动化生命周期管理 ---
@app.on_event("startup")
async def startup():
    # 启动时自动检查并创建数据库表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- 4. 异步持久化逻辑 ---
async def save_chat_to_db(prompt: str):
    async with AsyncSessionLocal() as session:
        new_log = ChatLog(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            prompt=prompt,
            response_summary="Streaming response logged." 
        )
        session.add(new_log)
        await session.commit()

# --- 5. 请求模型 ---
class ChatRequest(BaseModel):
    prompt: str
    image_data: Optional[str] = None
    history: Optional[List[dict]] = []

# --- 6. 核心接口 ---
@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest = Body(...), 
    background_tasks: BackgroundTasks = None
):
    # 将持久化操作放入后台任务，实现“即刻响应”
    if background_tasks:
        background_tasks.add_task(save_chat_to_db, request.prompt)
    
    return StreamingResponse(
        get_agent_response(request.prompt, request.image_data, request.history), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    # 运行在 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
