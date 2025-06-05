"""
Milvus连接管理器
"""

import time
import threading
from typing import Optional
from pymilvus import connections, utility
from src.config import MILVUS_HOST, MILVUS_PORT
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MilvusConnectionManager:
    """Milvus连接管理器，提供智能重连机制"""
    
    def __init__(
        self, 
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        alias: str = "default",
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        """初始化连接管理器
        
        Args:
            host: Milvus服务器主机
            port: Milvus服务器端口  
            alias: 连接别名
            max_retries: 最大重试次数
            initial_delay: 初始重试延迟（秒）
            max_delay: 最大重试延迟（秒）
            backoff_factor: 退避因子
        """
        self.host = host
        self.port = port
        self.alias = alias
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        self._connected = False
        self._lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 30  # 健康检查间隔（秒）
        
        logger.info(f"MilvusConnectionManager初始化: {host}:{port}")
    
    def connect(self) -> bool:
        """建立连接，带重试机制
        
        Returns:
            bool: 连接是否成功
        """
        with self._lock:
            if self._connected and self._check_connection():
                return True
            
            return self._connect_with_retry()
    
    def _connect_with_retry(self) -> bool:
        """带重试机制的连接方法"""
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                # 尝试连接
                connections.connect(
                    alias=self.alias,
                    host=self.host,
                    port=self.port
                )
                
                # 验证连接
                if self._verify_connection():
                    self._connected = True
                    logger.info(f"Milvus连接成功: {self.host}:{self.port} (尝试 {attempt + 1})")
                    return True
                    
            except Exception as e:
                logger.warning(f"连接尝试 {attempt + 1}/{self.max_retries + 1} 失败: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logger.error(f"达到最大重试次数，连接失败")
        
        self._connected = False
        return False
    
    def _verify_connection(self) -> bool:
        """验证连接是否有效"""
        try:
            # 尝试列出集合来验证连接
            utility.list_collections(using=self.alias)
            return True
        except Exception as e:
            logger.debug(f"连接验证失败: {e}")
            return False
    
    def is_connected(self) -> bool:
        """检查连接状态
        
        Returns:
            bool: 连接是否正常
        """
        with self._lock:
            current_time = time.time()
            
            # 如果距离上次检查超过检查间隔，重新检查
            if current_time - self._last_check_time > self._check_interval:
                self._last_check_time = current_time
                self._connected = self._check_connection()
            
            return self._connected
    
    def _check_connection(self) -> bool:
        """检查连接健康状态"""
        if not self._connected:
            return False
            
        try:
            # 尝试简单操作来检查连接
            utility.list_collections(using=self.alias)
            return True
        except Exception as e:
            logger.warning(f"连接健康检查失败: {e}")
            return False
    
    def reconnect(self) -> bool:
        """强制重连
        
        Returns:
            bool: 重连是否成功
        """
        logger.info("执行强制重连...")
        with self._lock:
            self._connected = False
            try:
                connections.disconnect(alias=self.alias)
            except:
                pass  # 忽略断开连接时的错误
            
            return self._connect_with_retry()
    
    def close(self):
        """关闭连接"""
        with self._lock:
            try:
                if self._connected:
                    connections.disconnect(alias=self.alias)
                    logger.info(f"Milvus连接已关闭: {self.alias}")
            except Exception as e:
                logger.warning(f"关闭连接时出错: {e}")
            finally:
                self._connected = False
    
    def get_connection_info(self) -> dict:
        """获取连接信息
        
        Returns:
            dict: 连接信息
        """
        return {
            "host": self.host,
            "port": self.port,
            "alias": self.alias,
            "connected": self._connected,
            "max_retries": self.max_retries,
            "last_check_time": self._last_check_time
        }

# 全局连接管理器单例
_connection_manager: Optional[MilvusConnectionManager] = None
_manager_lock = threading.Lock()

def get_connection_manager() -> MilvusConnectionManager:
    """获取全局连接管理器单例"""
    global _connection_manager
    
    if _connection_manager is None:
        with _manager_lock:
            if _connection_manager is None:
                _connection_manager = MilvusConnectionManager()
    
    return _connection_manager

def ensure_connection() -> bool:
    """确保Milvus连接可用
    
    Returns:
        bool: 连接是否可用
    """
    manager = get_connection_manager()
    return manager.connect() 