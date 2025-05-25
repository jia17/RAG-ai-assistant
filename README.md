#  AI 助手

基于 LangGraph 和 RAG（检索增强生成）技术构建的 智能问答助手，能够回答相关的技术问题，支持多轮对话和上下文记忆。

##  核心功能

- **智能问答**：基于知识库的专业技术问答
- **多轮对话**：支持上下文记忆的连续对话
- **自适应检索**：混合使用向量检索和关键词检索
- **答案评估**：自动评估答案质量并优化
- **查询重写**：智能重写用户查询以提高检索效果
- **Web 搜索增强**：当知识库无法回答时，自动进行网络搜索
- **持久化存储**：支持对话历史和状态的数据库持久化

## 核心架构

项目采用 LangGraph 状态机架构，通过多个节点协作完成智能问答：

用户查询 → 上下文管理 → 查询分析 → 文档检索 → 内容过滤 → 答案生成 → 质量评估 → 结果输出
                ↑                                                                    									   ↓
                └─────────── 查询重写/Web搜索 ←───────────────────────────┘



### 2. 环境变量配置

```bash
# LLM API Keys
QWEN_API_KEY=your_qwen_key

# LangChain 配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_key
```

### 3. 运行方式

```bash
# 交互式聊天模式
python main.py --interactive

# 默认模式（包含测试）
python main.py
```

## 技术栈

- **LangGraph** - 工作流编排和状态管理
- **LangChain** - LLM 应用开发框架
- **Milvus** - 向量数据库
- **SQLAlchemy** - 数据库 ORM
- **Sentence Transformers** - 文本嵌入

## 工作流程

1. **上下文管理** - 加载对话历史，管理会话状态
2. **查询分析** - 分析用户意图，提取关键实体
3. **文档检索** - 使用混合检索策略获取相关文档
4. **内容过滤** - 验证和过滤检索结果的相关性
5. **答案生成** - 基于过滤后的内容生成回答
6. **质量评估** - 评估答案质量，决定是否需要优化
7. **迭代优化** - 根据评估结果进行查询重写或 Web 搜索

## 📝目录结构

```markdown
RAG-ai-assistant/
│
├── README.md                      
├── requirements.txt               # 项目依赖
├── .env                   		   # 环境变量
├── .gitignore                   
├── main.py                        # 项目的命令行入口点
│
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── app.py                     # LangGraph 应用的主要构建和编译逻辑 (StateGraph 定义)
│   │
│   ├── config/                    # 配置文件
│   │   ├── __init__.py
│   │   └── settings.py            # 全局设置
│   │
│   ├── langgraph/                 # LangGraph状态机实现
│   │   ├── __init__.py
│   │   ├── states.py              # 状态定义
│   │   ├── nodes/                 # 节点实现
│   │   │   ├── __init__.py
│   │   │   ├── query_analyser.py         # 查询分析与预处理器
│   │   │   ├── adaptive_retriever.py     # 自适应混合检索器
│   │   │   ├── filter_validator.py       # 过滤与验证器
│   │   │   ├── generator.py              # 答案生成器
│   │   │   ├── answer_critique.py        # 答案评估与校验器
│   │   │   ├── query_rewriter.py         # 查询重写器
│   │   │   ├── web_search_node.py        # Web 搜索增强器
│   │   │   └── context_manager_node.py   # 上下文管理器
│   │   │
│   │   └── graph.py               # 状态图定义与边逻辑
│   │
│   ├── models/                    # 模型接口
│   │   ├── __init__.py
│   │   ├── llm_service.py            # 封装大语言模型调用 (如 Claude, 或本地模型)
│   │   ├── embedding_service.py      # 封装嵌入模型调用
│   │   └── vector_store_service.py   # 封装与 Milvus的交互
│   │
│   ├── knowledge_base/            # 知识库相关
│   │   ├── __init__.py
│   │   ├── data_processor.py      # 数据预处理
│   │   ├── markdown_splitter.py   # Markdown文档切分
│   │   └── vector_store.py        # 向量数据库接口
│   │
│   ├── storage/                   # 存储相关，持久化存储checkpoint
│   │   ├── __init__.py
│   │   ├── models.py              # 数据模型定义
│   │   ├── database.py            # 数据库操作
│   │   └── checkpointer.py        # 检查点管理
│   │
│   ├── prompts/                      # 存放 Prompt 模板
│   │   ├── __init__.py
│   │   ├── system_prompts.py         # 定义核心系统提示，如  AI 助手角色
│   │   └── task_prompts.py           # 定义特定任务的提示，如查询分析、答案评估等
│   │
│   ├── utils/                     # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py              # 日志工具
│   │   └── helpers.py             # 辅助函数
│   │
│   └── api/                       # API接口
│       ├── __init__.py
│       └── endpoints.py           # API端点
│
├── scripts/                       # 脚本目录
│   ├── data_ingestion/               # 数据获取与预处理脚本
│   │   ├── __init__.py
│   │   ├── fetch__docs.py  # 脚本：获取文档
│   ├── vectorization/
│   │   ├── __init__.py
│   │   └── vectorize_and_store.py    # 脚本：文本向量化并存入 Milvus
│   └── setup_milvus.py               # (可选) 脚本：帮助初始化 Milvus Schema 和索引
│
├── tests/                         # 测试目录
│   ├── __init__.py
│   ├── unit/                         # 单元测试
│   │   ├── __init__.py
│   │   ├── models/                   # 模型相关测试
│   │   └── knowladge_base/           # 知识库相关测试
│   └── integration/                  # 集成测试
│       ├── __init__.py
│       └── test_rag_pipeline.py
│
└── notebooks/                        # Jupyter Notebooks 用于实验、数据探索、模型评估
    ├── 01_data_exploration.ipynb     # 探索知识源
    ├── 02_embedding_testing.ipynb    # 测试不同嵌入模型的效果
    ├── 03_prompt_engineering.ipynb   # Prompt调优实验
    └── 04_rag_evaluation.ipynb       # RAG流程评估
```