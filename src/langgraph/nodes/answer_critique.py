"""
答案评估节点
"""
from typing import Dict, Any
import json
from src.models import default_llm
from src.prompts import ANSWER_EVALUATION_PROMPT, SYSTEM_PROMPT
from src.utils.logger import get_logger
from ..states import KubeSphereAgentState

logger = get_logger(__name__)

def evaluate_answer(state: KubeSphereAgentState) -> Dict[str, Any]:
    """
    评估生成的答案质量
    
    Args:
        state: 当前状态
        
    Returns:
        要更新到状态的字典
    """
    logger.info("评估答案质量...")
    
    # 获取必要的输入
    original_query = state["original_query"]
    
    # 获取生成的答案
    generation = state.get("generation", {})
    generated_answer = generation.get("answer", "")
    if not generated_answer and "answer" in state:
        generated_answer = state["answer"]
    
    # 获取过滤后的文档块
    filtered_chunks = state.get("filtering", {}).get("filtered_chunks", [])
    
    # 准备评估提示
    prompt = ANSWER_EVALUATION_PROMPT.format(
        original_query=original_query,
        filtered_chunks=json.dumps(filtered_chunks, ensure_ascii=False),
        generated_answer=generated_answer
    )
    
    try:
        # 调用LLM进行评估
        system_prompt = SYSTEM_PROMPT
        #TODO:待优化
        response = default_llm.generate(
            system_prompt=system_prompt,
            prompt=prompt
        )
        
        # 提取评估结果
        critique_text = response
        if isinstance(response, dict) and "text" in response:
            critique_text = response["text"]
        
        # 解析JSON评估结果
        try:
            # 尝试从文本中提取JSON
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            import re
            matches = re.findall(json_pattern, critique_text)
            
            if matches:
                critique_result = json.loads(matches[0])
            else:
                critique_result = json.loads(critique_text)
                
        except json.JSONDecodeError:
            logger.warning(f"答案评估结果非JSON格式: {critique_text}")
            # 如果无法解析JSON，创建默认评估结果
            critique_result = {
                "decision": "Accept", 
                "reasoning": "无法解析评估结果，默认接受",
                "score": 0.7
            }
        
        logger.info(f"答案评估结果: {critique_result}")
        
        # 获取工作流状态
        workflow = state.get("workflow", {})
        current_iteration = workflow.get("iteration_count", 0)
        rewrite_count = workflow.get("query_rewrite_count", 0)
        
        # 判断是否需要进行Web搜索
        needs_web_search = critique_result.get("decision") == "Reject_WebSearch"
        
        needs_improvement = critique_result.get("decision") != "Accept",
        decision = critique_result.get("decision")

        # 更新工作流状态
        updated_workflow = {
            "iteration_count": current_iteration,
            "query_rewrite_count": rewrite_count,
            "needs_web_search": needs_web_search
        }
        
        # 构建评估状态
        evaluation_state = {
            "query": original_query,
            "answer": generated_answer,
            "filtered_chunks": filtered_chunks,
            # "evaluation_scores": {
            #     "factuality": critique_result.get("factuality", 0.0),
            #     "relevance": critique_result.get("relevance", 0.0),
            #     "completeness": critique_result.get("completeness", 0.0),
            #     "clarity": critique_result.get("clarity", 0.0)
            # },
            "needs_improvement": needs_improvement,
            "decision": decision,
        }
        
        # 返回要更新的状态部分
        return {
            "evaluation": evaluation_state,
            "workflow": updated_workflow,
        }
    except Exception as e:
        logger.error(f"答案评估失败: {str(e)}")
        
        # 返回错误信息
        return {
            "evaluation": {
                "query": original_query,
                "answer": generated_answer,
                "needs_improvement": "False",
            },
            "error_message": f"答案评估失败: {str(e)}"
        }
