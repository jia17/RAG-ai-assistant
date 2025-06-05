"""
Milvus搜索参数优化器
"""

import math
from typing import Dict, Any, Optional
from pymilvus import Collection, utility
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SearchOptimizer:
    """Milvus搜索参数优化器"""
    
    def __init__(self):
        """初始化搜索优化器"""
        # 预定义的参数配置
        self.small_dataset_threshold = 10000      # 小数据集阈值
        self.medium_dataset_threshold = 100000    # 中等数据集阈值
        
        # 距离度量类型
        self.distance_metrics = {
            "COSINE": "COSINE",
            "L2": "L2", 
            "IP": "IP"
        }
        
        logger.info("SearchOptimizer初始化完成")
    
    def get_optimal_search_params(
        self, 
        collection: Collection,
        metric_type: str = "COSINE",
        consistency_level: str = "Strong"
    ) -> Dict[str, Any]:
        """获取最优搜索参数
        
        Args:
            collection: Milvus集合对象
            metric_type: 距离度量类型
            consistency_level: 一致性级别
            
        Returns:
            Dict[str, Any]: 优化的搜索参数
        """
        try:
            # 获取数据量
            data_count = self._get_collection_count(collection)
            logger.info(f"集合数据量: {data_count}")
            
            # 计算最优参数
            nprobe = self._calculate_optimal_nprobe(data_count)
            
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": nprobe},
                "consistency_level": consistency_level
            }
            
            logger.info(f"优化的搜索参数: {search_params}")
            return search_params
            
        except Exception as e:
            logger.error(f"获取搜索参数失败: {e}")
            # 返回默认参数
            return self._get_default_search_params(metric_type)
    
    def get_optimal_index_params(
        self, 
        data_count: Optional[int] = None,
        metric_type: str = "COSINE"
    ) -> Dict[str, Any]:
        """获取最优索引参数
        
        Args:
            data_count: 数据量，如果未提供则使用默认中等规模参数
            metric_type: 距离度量类型
            
        Returns:
            Dict[str, Any]: 优化的索引参数
        """
        try:
            if data_count is None:
                data_count = self.medium_dataset_threshold
            
            # 计算最优nlist
            nlist = self._calculate_optimal_nlist(data_count)
            
            index_params = {
                "metric_type": metric_type,
                "index_type": "IVF_FLAT",
                "params": {"nlist": nlist}
            }
            
            logger.info(f"优化的索引参数 (数据量: {data_count}): {index_params}")
            return index_params
            
        except Exception as e:
            logger.error(f"获取索引参数失败: {e}")
            return self._get_default_index_params(metric_type)
    
    def _calculate_optimal_nlist(self, data_count: int) -> int:
        """计算最优nlist参数
        
        Args:
            data_count: 数据量
            
        Returns:
            int: 最优nlist值
        """
        if data_count <= self.small_dataset_threshold:
            # 小数据集：nlist = 64
            nlist = 64
        elif data_count <= self.medium_dataset_threshold:
            # 中等数据集：nlist = 1024
            nlist = 1024
        else:
            # 大数据集：nlist = 4 * sqrt(data_count)
            nlist = min(4096, max(1024, int(4 * math.sqrt(data_count))))
        
        logger.debug(f"数据量 {data_count} 计算得到 nlist: {nlist}")
        return nlist
    
    def _calculate_optimal_nprobe(self, data_count: int) -> int:
        """计算最优nprobe参数
        
        Args:
            data_count: 数据量
            
        Returns:
            int: 最优nprobe值
        """
        if data_count <= self.small_dataset_threshold:
            # 小数据集：nprobe = 8 (nlist的12.5%)
            nprobe = 8
        elif data_count <= self.medium_dataset_threshold:
            # 中等数据集：nprobe = 64 (nlist的6.25%)
            nprobe = 64
        else:
            # 大数据集：动态计算，约为nlist的5%-10%
            nlist = self._calculate_optimal_nlist(data_count)
            nprobe = max(16, min(256, int(nlist * 0.08)))
        
        logger.debug(f"数据量 {data_count} 计算得到 nprobe: {nprobe}")
        return nprobe
    
    def _get_collection_count(self, collection: Collection) -> int:
        """获取集合中的数据量
        
        Args:
            collection: Milvus集合对象
            
        Returns:
            int: 数据量
        """
        try:
            collection.load()
            stats = collection.get_stats()
            
            # 从统计信息中提取行数
            for stat in stats:
                if stat.get("name") == "row_count":
                    return int(stat.get("value", 0))
            
            # 如果无法从统计信息获取，使用查询方式
            return collection.num_entities
            
        except Exception as e:
            logger.warning(f"获取集合数据量失败: {e}")
            return self.medium_dataset_threshold  # 返回默认值
    
    def _get_default_search_params(self, metric_type: str = "COSINE") -> Dict[str, Any]:
        """获取默认搜索参数
        
        Args:
            metric_type: 距离度量类型
            
        Returns:
            Dict[str, Any]: 默认搜索参数
        """
        return {
            "metric_type": metric_type,
            "params": {"nprobe": 64},
            "consistency_level": "Strong"
        }
    
    def _get_default_index_params(self, metric_type: str = "COSINE") -> Dict[str, Any]:
        """获取默认索引参数
        
        Args:
            metric_type: 距离度量类型
            
        Returns:
            Dict[str, Any]: 默认索引参数
        """
        return {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    
    def analyze_search_performance(
        self, 
        search_time: float,
        result_count: int,
        query_limit: int
    ) -> Dict[str, Any]:
        """分析搜索性能
        
        Args:
            search_time: 搜索耗时（秒）
            result_count: 返回结果数量
            query_limit: 查询限制数量
            
        Returns:
            Dict[str, Any]: 性能分析结果
        """
        performance_analysis = {
            "search_time": search_time,
            "result_count": result_count,
            "query_limit": query_limit,
            "qps": 1.0 / search_time if search_time > 0 else 0,
            "recall_rate": result_count / query_limit if query_limit > 0 else 0,
            "performance_level": "unknown"
        }
        
        # 性能等级判断
        if search_time < 0.1:
            performance_analysis["performance_level"] = "excellent"
        elif search_time < 0.5:
            performance_analysis["performance_level"] = "good"
        elif search_time < 2.0:
            performance_analysis["performance_level"] = "acceptable"
        else:
            performance_analysis["performance_level"] = "poor"
        
        return performance_analysis
    
    def suggest_optimization(
        self, 
        performance_analysis: Dict[str, Any],
        current_params: Dict[str, Any]
    ) -> Dict[str, str]:
        """基于性能分析建议优化策略
        
        Args:
            performance_analysis: 性能分析结果
            current_params: 当前搜索参数
            
        Returns:
            Dict[str, str]: 优化建议
        """
        suggestions = {}
        
        performance_level = performance_analysis.get("performance_level", "unknown")
        search_time = performance_analysis.get("search_time", 0)
        current_nprobe = current_params.get("params", {}).get("nprobe", 64)
        
        if performance_level == "poor":
            if search_time > 2.0:
                suggestions["nprobe"] = f"当前nprobe={current_nprobe}过高，建议降低到{max(16, current_nprobe // 2)}"
                suggestions["index"] = "考虑使用HNSW索引以获得更好性能"
        elif performance_level == "excellent":
            if performance_analysis.get("recall_rate", 0) < 0.9:
                suggestions["nprobe"] = f"当前nprobe={current_nprobe}可能过低，建议增加到{min(256, current_nprobe * 2)}"
        
        return suggestions 