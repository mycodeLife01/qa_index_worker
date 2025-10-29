"""
FastAPI 服务：提供文档索引检查和索引任务接口
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
from worker import IndexWorker
from config import load_config
from datetime import datetime


app = FastAPI(title="QA Index Worker API", version="1.0.0")

# 全局变量：配置和worker实例
config = None
worker = None


class CheckRequest(BaseModel):
    """检查文档是否已索引的请求"""
    content_hash: str


class CheckResponse(BaseModel):
    """检查文档的响应"""
    content_hash: str
    exists: bool
    chunk_count: Optional[int] = None
    indexed_at: Optional[str] = None


class IndexRequest(BaseModel):
    """索引文档的请求"""
    job_id: str
    content_hash: str
    file_url: str
    file_type: str


class IndexResponse(BaseModel):
    """索引文档的响应"""
    job_id: str
    status: str  # "success" or "failed"
    message: str
    chunk_count: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化配置和worker"""
    global config, worker
    config = load_config()
    worker = IndexWorker(config)
    print("[API] Index Worker API started successfully")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "qa-index-worker",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/check", response_model=CheckResponse)
async def check_document(request: CheckRequest):
    """
    检查文档是否已经被索引
    
    Args:
        request: 包含 content_hash 的请求
        
    Returns:
        CheckResponse: 包含文档索引状态的响应
    """
    try:
        # 查询向量数据库中是否存在该 content_hash 的文档
        results = worker.vector_store.get(
            where={"content_hash": request.content_hash}
        )
        
        exists = len(results.get("ids", [])) > 0
        chunk_count = len(results.get("ids", [])) if exists else None
        
        # 获取最新的索引时间
        indexed_at = None
        if exists and results.get("metadatas"):
            # 从所有chunks中获取最新的indexed_at时间
            metadatas = results.get("metadatas", [])
            if metadatas and metadatas[0].get("indexed_at"):
                indexed_at = metadatas[0]["indexed_at"]
        
        return CheckResponse(
            content_hash=request.content_hash,
            exists=exists,
            chunk_count=chunk_count,
            indexed_at=indexed_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查文档失败: {str(e)}")


@app.post("/index", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    """
    执行文档索引任务
    
    Args:
        request: 包含任务信息的请求
        
    Returns:
        IndexResponse: 索引任务的结果
    """
    try:
        print(f"[API] 收到索引请求: job_id={request.job_id}, content_hash={request.content_hash}")
        
        # 构造任务数据
        job_data = {
            "content_hash": request.content_hash,
            "file_url": request.file_url,
            "file_type": request.file_type,
        }
        
        # 执行索引任务
        await worker.run(job_id=request.job_id, job_data=job_data)
        
        # 查询索引后的chunk数量
        results = worker.vector_store.get(
            where={"content_hash": request.content_hash}
        )
        chunk_count = len(results.get("ids", []))
        
        print(f"[API] 索引完成: job_id={request.job_id}, chunks={chunk_count}")
        
        return IndexResponse(
            job_id=request.job_id,
            status="success",
            message=f"文档索引成功，共生成 {chunk_count} 个语义块",
            chunk_count=chunk_count
        )
    except Exception as e:
        error_msg = f"索引文档失败: {str(e)}"
        print(f"[API] 错误: {error_msg}")
        return IndexResponse(
            job_id=request.job_id,
            status="failed",
            message=error_msg,
            chunk_count=None
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

