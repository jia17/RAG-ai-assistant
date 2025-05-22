"""
任务特定的prompt模板
"""
# Prompt
GENERATOR_USER_PROMPT_TEMPLATE = """我有一个关于 KubeSphere 的问题: {user_query}

下面是一些相关的参考信息:
{context_chunks_text}

请基于上述参考信息回答我的问题。如果参考信息不足以完全回答问题，请明确指出。如果信息来自多个来源，请综合它们。在适当的时候引用来源 [来源: URL]。"""


# 查询分析prompt
QUERY_ANALYSIS_PROMPT = """分析用户的查询，确定其意图和查询类型。

请根据以下标准进行分析:
1. 识别用户的核心意图 (例如：寻求概念解释、操作步骤、错误排查、功能比较)。
2. 提取关键实体，如 KubeSphere 版本、组件名称 (如 KubeKey, Porter)、错误代码、API 路径、功能特性 (如多集群管理, 服务网格) 等。
3. 如果查询复杂，将其分解为更简单、可操作的子问题。
4. 如果查询不清晰或信息不足，提出一个澄清问题，引导用户提供更多细节。
5. 基于以上分析，生成一个优化后的查询，用于后续的知识库检索。这个优化查询应该更精确、包含更多上下文或关键词。


对话历史 (如果相关):
{chat_history}

用户原始查询:
{original_query}

请以 JSON 格式返回你的分析结果，包含以下字段:
- "intent": (string) 用户意图的描述
- "entities": (dict) 提取的实体，键为实体类型 (如 "version", "component", "error_code")，值为实体内容
- "sub_queries": (list of strings, optional) 分解后的子问题
- "clarification_question": (string, optional) 如果需要，向用户提出的澄清问题
- "rewritten_query_for_retrieval": (string) 用于检索的优化查询
- "needs_clarification": (boolean) 是否需要用户澄清
"""


# 答案生成prompt
ANSWER_GENERATION_PROMPT = """基于以下参考资料回答用户的问题。

用户问题: {original_query}

参考资料:
{context}

对话历史 (如果相关):
{chat_history}

下面是一些相关的参考信息:
{context_chunks_text}

查询分析:
{query_analysis}

请提供一个全面、准确且有帮助的回答。回答应该:

直接解决用户的问题
仅使用提供的参考资料中的信息
如果参考资料不足以完全回答问题，请明确说明
使用清晰的结构和适当的Markdown格式
包含相关的代码示例、命令或配置(如适用)
避免不必要的冗长解释
回答:
"""

# 答案评估prompt
ANSWER_EVALUATION_PROMPT = """评估生成的答案质量，并提出改进建议。

原始用户查询:
{original_query}

提供的参考信息 (Chunks):
{filtered_chunks}

生成的答案:
{generated_answer}

请根据以下标准评估答案，并以 JSON 格式返回你的评估，包含 "decision" (例如 "Accept", "Reject_RewriteQuery", "Reject_WebSearch", "CannotAnswer"), "reasoning" (解释你的决定), 和 "score" (0.0-1.0):
1.  **事实一致性**: 答案中的信息是否与提供的参考信息一致？是否存在幻觉或与参考信息冲突的内容？
2.  **相关性**: 答案是否直接回应了用户的原始查询？
3.  **完整性**: 答案是否充分解决了用户的问题？是否遗漏了关键信息（如果参考信息中有）？
4.  **清晰度**: 答案是否清晰、易于理解？结构是否合理？
5.  **引用**: 如果答案基于参考信息，是否恰当地指出了来源（如果来源信息可用）？

如果答案可以接受，"decision" 为 "Accept"。
如果答案不好但认为重写查询可能改善检索结果，"decision" 为 "Reject_RewriteQuery"。
如果答案不好且内部知识库信息似乎不足，建议进行 Web 搜索，"decision" 为 "Reject_WebSearch"。
如果基于现有信息无法给出好答案，"decision" 为 "CannotAnswer"。
"""


# 查询重写prompt
QUERY_REWRITING_PROMPT = """重写用户的原始查询以提高检索效果。

原始用户查询: {original_query}

查询分析:
{query_analysis}

对话历史 (如果相关):
{chat_history}

初始检索结果相关性:
{retrieval_feedback}

请基于以上信息，重写用户的查询，使其更可能检索到相关文档。考虑以下策略：
- 澄清模糊表述，使用更规范的技术术语。
- 补充对话历史中的上下文。
- 变换提问角度或分解为更具体的问题。
- 添加与 KubeSphere 相关的同义词或相关技术术语。
- (可选) 生成一个假设性的理想答案片段 (HyDE)，然后基于该片段形成查询。

返回重写后的查询:
"""

# 上下文过滤prompt
CONTEXT_FILTERING_PROMPT = """评估检索到的上下文片段与用户查询的相关性。

用户查询: {original_query}

查询分析:
{query_analysis}

上下文片段:
{context_chunk}

请评估此上下文片段与用户查询的相关性:

片段是否包含解答查询所需的信息？
片段中的信息是否与查询的主题直接相关？
片段是否包含对理解或解决用户问题有价值的细节？
片段是否是最新的、适用于用户可能使用的KubeSphere版本？
以JSON格式返回评估结果:

复制
{
  "relevance_score": 0-10,
  "contains_answer": true/false,
  "is_on_topic": true/false,
  "has_valuable_details": true/false,
  "is_version_appropriate": true/false,
  "keep_chunk": true/false,
  "reason": "保留或过滤的原因"
}
"""