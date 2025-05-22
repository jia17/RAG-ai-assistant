"""
API端点定义
"""

from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
import uvicorn
from src.app import process_query, create_app
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="KubeSphere AI Assistant API",
    description="KubeSphere社区问答AI助手API",
    version="0.1.0"
)

# 定义请求和响应模型
class QueryRequest(BaseModel):
    query: str
    include_state: bool = False

class QueryResponse(BaseModel):
    query: str
    answer: str
    state: Optional[Dict[str, Any]] = None

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> Dict[str, Any]:
    """处理查询请求
    
    Args:
        request: 包含查询文本的请求对象
        
    Returns:
        包含答案的响应对象
    """
    try:
        result = process_query(request.query)
        
        # 根据请求决定是否包含状态
        if not request.include_state:
            result.pop("state", None)
            
        return result
    except Exception as e:
        logger.error(f"处理查询时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")



def start_api_server(host: str = "127.0.0.1", port: int = 8000):
    """启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api_server()
