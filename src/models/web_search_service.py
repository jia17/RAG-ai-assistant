"""
Web搜索服务封装
实现Tavily API集成、查询优化和智能参数调整
"""

import os
import re
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from src.config.settings import web_search_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WebSearchError(Exception):
    """Web搜索相关异常基类"""
    pass

class QuotaExhaustedError(WebSearchError):
    """API配额耗尽异常"""
    pass

class NetworkTimeoutError(WebSearchError):
    """网络超时异常"""
    pass

class LowQualityResultsError(WebSearchError):
    """搜索结果质量低异常"""
    pass

class InvalidAPIKeyError(WebSearchError):
    """API密钥无效异常"""
    pass

class QueryOptimizer:
    """查询优化器 - 负责分析查询类型和生成优化参数"""
    
    # 简单查询关键词模式
    SIMPLE_PATTERNS = [
        r'^什么是\s*\w+\?*$',  # "什么是Kubernetes?"
        r'^.{1,20}\s*是什么\?*$',  # "Docker是什么?"
        r'^\w+\s*定义\?*$',  # "容器定义"
        r'^如何\s*\w+\?*$',  # "如何安装"
    ]
    
    # 复杂查询关键词模式
    COMPLEX_PATTERNS = [
        r'如何.*配置.*',  # 配置相关
        r'.*故障.*排查.*',  # 故障排查
        r'.*错误.*解决.*',  # 错误解决
        r'.*最佳实践.*',  # 最佳实践
        r'.*性能.*优化.*',  # 性能优化
    ]
    
    # 紧急查询关键词
    URGENT_KEYWORDS = ['故障', '错误', '失败', '宕机', '崩溃', '紧急', '无法访问', 'error', 'failed', 'crash']
    
    # 中文技术术语映射
    CHINESE_TECH_TERMS = {
        '容器': 'container',
        '集群': 'cluster',
        '服务网格': 'service mesh',
        '微服务': 'microservices',
        '负载均衡': 'load balancer',
        '自动伸缩': 'auto scaling',
        '持续集成': 'continuous integration',
        '持续部署': 'continuous deployment',
    }
    
    @classmethod
    def analyze_query_complexity(cls, query: str) -> str:
        """分析查询复杂度"""
        query = query.strip()
        
        # 检查紧急关键词
        for keyword in cls.URGENT_KEYWORDS:
            if keyword in query.lower():
                return "urgent"
        
        # 检查简单模式
        for pattern in cls.SIMPLE_PATTERNS:
            if re.match(pattern, query):
                return "simple"
        
        # 检查复杂模式
        for pattern in cls.COMPLEX_PATTERNS:
            if re.search(pattern, query):
                return "complex"
        
        # 根据查询长度判断
        if len(query) < 20:
            return "simple"
        elif len(query) > 50:
            return "complex"
        
        return "simple"
    
    @classmethod
    def classify_query_intent(cls, query: str) -> str:
        """分类查询意图"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['什么是', '定义', 'what is', 'define']):
            return "definition"
        elif any(word in query_lower for word in ['如何', '怎么', 'how to', '步骤']):
            return "instruction"
        elif any(word in query_lower for word in ['故障', '错误', '问题', 'error', 'issue', 'problem']):
            return "troubleshooting"
        elif any(word in query_lower for word in ['比较', '区别', 'compare', 'difference']):
            return "comparison"
        else:
            return "general"
    
    @classmethod
    def is_chinese_query(cls, query: str) -> bool:
        """判断是否为中文查询"""
        chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', query))
        return chinese_char_count > len(query) * 0.3
    
    @classmethod
    def generate_search_params(cls, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """生成动态搜索参数"""
        complexity = cls.analyze_query_complexity(query)
        intent = cls.classify_query_intent(query)
        is_chinese = cls.is_chinese_query(query)
        
        # 获取基础参数
        params = web_search_config.get_search_params_for_query_type(complexity)
        
        # 添加时间范围
        if intent == "troubleshooting":
            params["time_range"] = "month"  # 故障排查优先最近的信息
        elif "最新" in query or "latest" in query.lower():
            params["time_range"] = "week"
        
        # 中文查询优化
        if is_chinese:
            params["include_domains"] = web_search_config.chinese_domains
            params["country"] = "CN"
        
        # 添加查询元数据
        params["query_metadata"] = {
            "complexity": complexity,
            "intent": intent,
            "is_chinese": is_chinese,
            "original_length": len(query)
        }
        
        return params
    
    # @classmethod
    # def optimize_query_for_search(cls, query: str) -> List[str]:
    #     """优化查询词用于搜索"""
    #     queries = [query]  # 原始查询
        
    #     # 中文查询添加英文翻译
    #     if cls.is_chinese_query(query):
    #         english_query = cls._translate_chinese_terms(query)
    #         if english_query != query:
    #             queries.append(english_query)
        
    #     return queries
    
    # @classmethod
    # def _translate_chinese_terms(cls, query: str) -> str:
    #     """翻译中文技术术语"""
    #     translated_query = query
    #     for chinese, english in cls.CHINESE_TECH_TERMS.items():
    #         translated_query = translated_query.replace(chinese, english)
    #     return translated_query

class WebSearchService:
    """Web搜索服务主类"""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """初始化Web搜索服务"""
        self.api_key = api_key or web_search_config.api_key
        self.timeout = timeout or web_search_config.timeout
        self.enabled = web_search_config.enabled and bool(self.api_key)
        
        # 初始化客户端
        self.client = None
        if self.enabled and TavilyClient:
            try:
                self.client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                self.enabled = False
        
        # 配额管理
        self.quota_used = 0
        self.quota_limit = web_search_config.quota_limit
        self.quota_reset_time = datetime.now() + timedelta(days=30)  # 月度重置
        
        # 缓存管理
        self.cache = {}
        self.cache_ttl = web_search_config.cache_ttl
        
        if not self.enabled:
            logger.warning("Web search service is disabled")
    
    def search(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行搜索请求"""
        if not self.enabled:
            raise WebSearchError("Web search service is disabled")
        
        if not self.client:
            raise WebSearchError("Tavily client is not initialized")
        
        # 检查配额
        if not self.check_quota():
            raise QuotaExhaustedError("API quota exhausted")
        
        # 生成搜索参数
        if params is None:
            params = QueryOptimizer.generate_search_params(query)
        
        # 检查缓存
        cache_key = self._generate_cache_key(query, params)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Using cached result for query: {query[:50]}...")
            return cached_result
        
        # 执行搜索
        try:
            result = self._execute_search_with_retry(query, params)
            
            # 验证结果质量
            if not self.validate_search_results(result.get("results", [])):
                raise LowQualityResultsError("Search results quality is too low")
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            # 更新配额使用
            self.quota_used += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _execute_search_with_retry(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """带重试机制的搜索执行"""
        max_retries = web_search_config.max_retries
        retry_delay = web_search_config.retry_delay
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # 准备Tavily搜索参数
                tavily_params = self._prepare_tavily_params(params)
                
                # 执行搜索
                start_time = time.time()
                response = self.client.search(query, **tavily_params)
                response_time = time.time() - start_time
                
                # 格式化响应
                formatted_result = self._format_search_result(response, response_time, params)
                
                logger.info(f"Search successful on attempt {attempt + 1}: {query[:50]}...")
                return formatted_result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # 指数退避
                else:
                    # 最后一次尝试失败，检查是否需要降级
                    if web_search_config.enable_fallback:
                        return self._handle_search_failure(query, params, e)
        
        # 所有重试都失败了
        raise last_exception or WebSearchError("All search attempts failed")
    
    def _prepare_tavily_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """准备Tavily API参数"""
        tavily_params = {}
        
        # 映射参数
        if "search_depth" in params:
            tavily_params["search_depth"] = params["search_depth"]
        if "max_results" in params:
            tavily_params["max_results"] = min(params["max_results"], 10)  # Tavily限制
        if "time_range" in params:
            tavily_params["time_range"] = params["time_range"]
        if "include_domains" in params:
            tavily_params["include_domains"] = params["include_domains"]
        if "country" in params:
            tavily_params["country"] = params["country"]
        
        # 默认参数
        tavily_params.setdefault("include_raw_content", False)
        tavily_params.setdefault("timeout", self.timeout)
        
        return tavily_params
    
    def _format_search_result(self, response: Dict, response_time: float, params: Dict[str, Any]) -> Dict[str, Any]:
        """格式化搜索结果"""
        return {
            "query": response.get("query", ""),
            "web_results": response.get("results", []),
            "web_search_success": True,
            "search_params": params,
            "search_metadata": {
                "response_time": response_time,
                "total_results": len(response.get("results", [])),
                "api_response_time": response.get("response_time", 0),
                "timestamp": datetime.now().isoformat()
            },
            "search_quality_score": self._calculate_quality_score(response.get("results", [])),
            "fallback_attempted": False
        }
    
    def _calculate_quality_score(self, results: List[Dict[str, Any]]) -> float:
        """计算搜索结果质量评分"""
        if not results:
            return 0.0
        
        total_score = 0.0
        for result in results:
            score = result.get("score", 0.5)  # Tavily提供的相关性评分
            content_length = len(result.get("content", ""))
            title_length = len(result.get("title", ""))
            
            # 综合评分：相关性 + 内容完整性
            content_score = min(content_length / 500, 1.0)  # 内容长度评分
            title_score = min(title_length / 50, 1.0)  # 标题评分
            
            total_score += (score * 0.6 + content_score * 0.3 + title_score * 0.1)
        
        return total_score / len(results)
    
    def _handle_search_failure(self, query: str, params: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """处理搜索失败的降级策略"""
        logger.warning(f"Implementing fallback strategy for query: {query[:50]}...")
        
        # 返回失败状态但不抛出异常
        return {
            "query": query,
            "web_results": [],
            "web_search_success": False,
            "search_params": params,
            "error_message": str(error),
            "search_metadata": {
                "timestamp": datetime.now().isoformat(),
                "fallback_reason": type(error).__name__
            },
            "search_quality_score": 0.0,
            "fallback_attempted": True
        }
    
    def validate_search_results(self, results: List[Dict[str, Any]]) -> bool:
        """验证搜索结果质量"""
        if not results:
            return False
        
        # 检查结果数量
        if len(results) < 2:
            return False
        
        # 检查内容质量
        valid_results = 0
        for result in results:
            content = result.get("content", "")
            title = result.get("title", "")
            
            if len(content) > 50 and len(title) > 5:
                valid_results += 1
        
        # 至少要有一半的结果是有效的
        return valid_results >= len(results) // 2
    
    def check_quota(self) -> bool:
        """检查API配额"""
        # 检查是否需要重置配额
        if datetime.now() > self.quota_reset_time:
            self.quota_used = 0
            self.quota_reset_time = datetime.now() + timedelta(days=30)
        
        return self.quota_used < self.quota_limit
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建包含查询和关键参数的哈希
        key_data = {
            "query": query.lower().strip(),
            "search_depth": params.get("search_depth", "basic"),
            "max_results": params.get("max_results", 5),
            "time_range": params.get("time_range"),
            "include_domains": params.get("include_domains")
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """缓存搜索结果"""
        self.cache[cache_key] = (result, datetime.now())
        
        # 简单的缓存清理：如果缓存太大，删除最旧的条目
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
    
    def get_quota_status(self) -> Dict[str, Any]:
        """获取配额使用状态"""
        return {
            "used": self.quota_used,
            "limit": self.quota_limit,
            "remaining": self.quota_limit - self.quota_used,
            "reset_time": self.quota_reset_time.isoformat(),
            "usage_percentage": (self.quota_used / self.quota_limit) * 100
        }

# 全局搜索服务实例
web_search_service = WebSearchService() 