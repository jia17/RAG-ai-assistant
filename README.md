# kubesphere-ai-assistant


```markdown
kubesphere-ai-assistant/
│
├── README.md                      # 项目说明文档
├── requirements.txt               # 项目依赖
├── .env.example                   # 环境变量示例
├── .gitignore                     # Git忽略文件
├── main.py                        # 项目的命令行入口点 (例如，启动一个交互式聊天或API服务)
│
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── app.py                     # LangGraph 应用的主要构建和编译逻辑 (StateGraph 定义)
│   │
│   ├── config/                    # 配置文件
│   │   ├── __init__.py
│   │   ├── settings.py            # 全局设置
│   │   └── prompts.py             # 提示模板
│   │
│   ├── langgraph/                 # LangGraph状态机实现
│   │   ├── __init__.py
│   │   ├── states.py              # 状态定义
│   │   ├── nodes/                 # 节点实现
│   │   │   ├── __init__.py
│   │   │   ├── query_analyser.py         # 查询分析与预处理器
│   │   │   ├── adaptive_retriever.py     # 自适应混合检索器 (或 retriever.py)
│   │   │   ├── filter_validator.py       # 过滤与验证器
│   │   │   ├── generator.py              # 答案生成器
│   │   │   ├── answer_critique.py        # 答案评估与校验器
│   │   │   ├── query_rewriter.py         # 查询重写器
│   │   │   ├── web_search_node.py        # Web 搜索增强器 (如果实现)
│   │   │   └── context_manager_node.py   # 上下文管理器 (PDF 3.2节末尾 & 6.1节)
│   │   │
│   │   └── graph.py               # 状态图定义与边逻辑
│   │
│   ├── models/                    # 模型接口
│   │   ├── __init__.py
│   │   ├── llm_service.py            # 封装大语言模型调用 (如 Claude, OpenAI API, 或本地模型)
│   │   ├── embedding_service.py      # 封装嵌入模型调用
│   │   └── vector_store_service.py   # 封装与 Milvus (或其他向量数据库) 的交互
│   │
│   ├── knowledge_base/            # 知识库相关
│   │   ├── __init__.py
│   │   ├── data_processor.py      # 数据预处理
│   │   ├── document_splitter.py   # 文档切分
│   │   └── vector_store.py        # 向量数据库接口
│   │
│   │
│   ├── prompts/                      # 存放 Prompt 模板 (对应PDF 2.2节 Prompt设计, 3.1节系统提示)
│   │   ├── __init__.py
│   │   ├── system_prompts.py         # 定义核心系统提示，如 KubeSphere AI 助手角色
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
│   │   ├── fetch_kubesphere_docs.py  # 脚本：获取 KubeSphere 官方文档
│   │   ├── fetch_github_issues.py    # 脚本：获取 GitHub Issues
│   │   ├── preprocess_text.py        # 脚本：文本清洗、结构化信息提取
│   │   └── chunk_documents.py        # 脚本：高级文档切分
│   ├── vectorization/
│   │   └── vectorize_and_store.py    # 脚本：文本向量化并存入 Milvus
│   └── setup_milvus.py               # (可选) 脚本：帮助初始化 Milvus Schema 和索引
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   ├── docs/                  # KubeSphere文档
│   │   ├── github/                # GitHub Issues/PRs
│   │   └── forum/                 # 社区论坛数据
│   │
│   └── processed/                 # 处理后的数据
│       └── chunks/                # 切分后的文档块
│
├── tests/                         # 测试目录
│   ├── __init__.py
│   ├── unit/                         # 单元测试
│   │   ├── test_nodes.py
│   │   └── test_services.py
│   └── integration/                  # 集成测试
│       └── test_rag_pipeline.py
│
├── notebooks/                        # Jupyter Notebooks 用于实验、数据探索、模型评估
│   ├── 01_data_exploration.ipynb     # 探索 KubeSphere 知识源
│   ├── 02_embedding_testing.ipynb    # 测试不同嵌入模型的效果
│   ├── 03_prompt_engineering.ipynb   # Prompt 调优实验
│   └── 04_rag_evaluation.ipynb       # RAG 流程评估
```



```markdown
需要配置env才能运行
env
# LLM API Keys
QWEN_API_KEY=qwen_key

# LangChain 配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lengchain_key

```