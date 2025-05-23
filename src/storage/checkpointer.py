"""
优化后的数据库检查点实现 - 简化版本
"""

import uuid
import base64
from typing import Dict, Any, Optional, Iterator, List, Tuple
from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .database import get_database_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseCheckpointer(BaseCheckpointSaver):
    """基于数据库的LangGraph检查点保存器 - 优化版本"""
    
    def __init__(self, database_url: Optional[str] = None):
        """初始化数据库检查点保存器"""
        super().__init__(serde=JsonPlusSerializer())
        self.db_manager = get_database_manager()
        logger.info("数据库检查点保存器初始化成功")
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """保存检查点"""
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                thread_id = str(uuid.uuid4())
                config.setdefault("configurable", {})["thread_id"] = thread_id
            
            # 使用 LangGraph 内置序列化器，转换为base64字符串存储
            checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
            metadata_type, metadata_bytes = self.serde.dumps_typed(metadata)
            
            # 将二进制数据编码为base64字符串
            checkpoint_data = {
                "type": checkpoint_type,
                "data": base64.b64encode(checkpoint_bytes).decode('utf-8')
            }
            metadata_data = {
                "type": metadata_type,
                "data": base64.b64encode(metadata_bytes).decode('utf-8')
            }
            
            # 安全地获取checkpoint的id和timestamp
            checkpoint_id = None
            checkpoint_ts = None
            
            if hasattr(checkpoint, 'id'):
                checkpoint_id = checkpoint.id
            elif isinstance(checkpoint, dict):
                checkpoint_id = checkpoint.get('id')
            
            if hasattr(checkpoint, 'ts'):
                checkpoint_ts = checkpoint.ts
            elif isinstance(checkpoint, dict):
                checkpoint_ts = checkpoint.get('ts')
            
            # 如果没有id或ts，生成默认值
            if not checkpoint_id:
                checkpoint_id = str(uuid.uuid4())
            if not checkpoint_ts:
                checkpoint_ts = datetime.now().isoformat()
            
            # 准备状态数据
            state_data = {
                "checkpoint": checkpoint_data,
                "metadata": metadata_data,
                "new_versions": new_versions,
                "checkpoint_id": checkpoint_id,
                "timestamp": checkpoint_ts
            }
            
            # 提取查询和答案用于索引
            current_query = self._extract_query(checkpoint)
            latest_answer = self._extract_answer(checkpoint)
            
            # 保存到数据库
            success = self.db_manager.save_conversation_state(
                conversation_id=thread_id,
                state_data=state_data,
                current_query=current_query,
                latest_answer=latest_answer
            )
            
            if not success:
                logger.warning(f"保存检查点失败: {thread_id}")
            
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": checkpoint_ts
                }
            }
            
        except Exception as e:
            logger.error(f"保存检查点时出错: {e}")
            return config
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """获取检查点元组"""
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return None
            
            # 从数据库加载状态
            state_data = self.db_manager.load_conversation_state(thread_id)
            if not state_data:
                return None
            
            # 反序列化检查点和元数据
            checkpoint = self._deserialize_checkpoint(state_data)
            metadata = self._deserialize_metadata(state_data)
            
            if not checkpoint:
                return None
            
            result_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": checkpoint.ts
                }
            }
            
            return CheckpointTuple(
                config=result_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None
            )
            
        except Exception as e:
            logger.error(f"获取检查点时出错: {e}")
            return None
    
    def list(
        self,
        config: Dict[str, Any],
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """列出检查点"""
        try:
            result = self.get_tuple(config)
            if result:
                yield result
        except Exception as e:
            logger.error(f"列出检查点时出错: {e}")
    
    def put_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str
    ) -> None:
        """保存写入操作"""
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return
            
            # 简化的写入操作保存
            write_data = {
                "task_id": task_id,
                "writes": writes,
                "timestamp": datetime.now().isoformat()
            }
            
            existing_state = self.db_manager.load_conversation_state(thread_id)
            if existing_state:
                existing_state.setdefault("pending_writes", []).append(write_data)
                self.db_manager.save_conversation_state(
                    conversation_id=thread_id,
                    state_data=existing_state
                )
            
        except Exception as e:
            logger.error(f"保存写入操作时出错: {e}")
    
    def get_next_version(self, current: Optional[str], channel: str) -> str:
        """获取下一个版本号"""
        try:
            if current is None:
                return "1"
            return str(int(current) + 1)
        except (ValueError, TypeError):
            return "1"
    
    def _extract_query(self, checkpoint: Checkpoint) -> Optional[str]:
        """提取查询信息"""
        try:
            # 安全地获取channel_values
            channel_values = None
            if hasattr(checkpoint, 'channel_values'):
                channel_values = checkpoint.channel_values
            elif isinstance(checkpoint, dict):
                channel_values = checkpoint.get('channel_values')
            
            if channel_values and isinstance(channel_values, dict):
                return channel_values.get("original_query")
            return None
        except Exception:
            return None
    
    def _extract_answer(self, checkpoint: Checkpoint) -> Optional[str]:
        """提取答案信息"""
        try:
            # 安全地获取channel_values
            channel_values = None
            if hasattr(checkpoint, 'channel_values'):
                channel_values = checkpoint.channel_values
            elif isinstance(checkpoint, dict):
                channel_values = checkpoint.get('channel_values')
            
            if not channel_values or not isinstance(channel_values, dict):
                return None
                
            generation = channel_values.get("generation", {})
            
            if isinstance(generation, dict):
                answer = generation.get("answer")
                if answer:
                    return answer
            
            return channel_values.get("answer")
        except Exception:
            return None
    
    def _deserialize_checkpoint(self, state_data: Dict[str, Any]) -> Optional[Checkpoint]:
        """反序列化检查点"""
        try:
            checkpoint_data = state_data.get("checkpoint")
            if not checkpoint_data:
                # 从原始状态创建基本检查点
                return Checkpoint(
                    v=1,
                    id=state_data.get("checkpoint_id", str(uuid.uuid4())),
                    ts=state_data.get("timestamp", datetime.now().isoformat()),
                    channel_values=state_data,
                    channel_versions={},
                    versions_seen={},
                    pending_sends=[]
                )
            
            # 处理新的base64格式
            if isinstance(checkpoint_data, dict) and "type" in checkpoint_data and "data" in checkpoint_data:
                try:
                    # 解码base64数据
                    checkpoint_bytes = base64.b64decode(checkpoint_data["data"])
                    # 使用 LangGraph 序列化器反序列化
                    return self.serde.loads_typed((checkpoint_data["type"], checkpoint_bytes))
                except Exception as e:
                    logger.warning(f"新格式反序列化失败: {e}")
            
            # 处理旧格式的兼容性（如果存在）
            if isinstance(checkpoint_data, (str, bytes)):
                if isinstance(checkpoint_data, str):
                    checkpoint_data = checkpoint_data.encode()
                try:
                    return self.serde.loads_typed(("checkpoint", checkpoint_data))
                except Exception as e:
                    logger.warning(f"旧格式反序列化失败: {e}")
            
            # 如果都失败，从原始状态创建基本检查点
            logger.warning("所有反序列化方法都失败，创建基本检查点")
            return Checkpoint(
                v=1,
                id=state_data.get("checkpoint_id", str(uuid.uuid4())),
                ts=state_data.get("timestamp", datetime.now().isoformat()),
                channel_values=state_data,
                channel_versions={},
                versions_seen={},
                pending_sends=[]
            )
            
        except Exception as e:
            logger.error(f"反序列化检查点失败: {e}")
            # 返回基本检查点作为后备
            return Checkpoint(
                v=1,
                id=str(uuid.uuid4()),
                ts=datetime.now().isoformat(),
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[]
            )
    
    def _deserialize_metadata(self, state_data: Dict[str, Any]) -> CheckpointMetadata:
        """反序列化元数据"""
        try:
            metadata_data = state_data.get("metadata")
            if not metadata_data:
                return CheckpointMetadata(
                    source="database",
                    step=0,
                    writes={},
                    parents={}
                )
            
            # 处理新的base64格式
            if isinstance(metadata_data, dict) and "type" in metadata_data and "data" in metadata_data:
                try:
                    # 解码base64数据
                    metadata_bytes = base64.b64decode(metadata_data["data"])
                    # 使用 LangGraph 序列化器反序列化
                    return self.serde.loads_typed((metadata_data["type"], metadata_bytes))
                except Exception as e:
                    logger.warning(f"新格式元数据反序列化失败: {e}")
            
            # 处理旧格式的兼容性（如果存在）
            if isinstance(metadata_data, (str, bytes)):
                if isinstance(metadata_data, str):
                    metadata_data = metadata_data.encode()
                try:
                    return self.serde.loads_typed(("metadata", metadata_data))
                except Exception as e:
                    logger.warning(f"旧格式元数据反序列化失败: {e}")
            
            # 如果都失败，返回默认元数据
            return CheckpointMetadata(
                source="database",
                step=0,
                writes={},
                parents={}
            )
            
        except Exception as e:
            logger.error(f"反序列化元数据失败: {e}")
            return CheckpointMetadata(
                source="database",
                step=0,
                writes={},
                parents={}
            )


# 全局检查点保存器实例
_checkpointer: Optional[DatabaseCheckpointer] = None


def get_database_checkpointer() -> DatabaseCheckpointer:
    """获取全局数据库检查点保存器实例"""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = DatabaseCheckpointer()
    return _checkpointer 