import argparse
import getpass
import os
import sys
from dotenv import load_dotenv


# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KubeSphere AI 助手命令行入口点
"""

# 加载环境变量
load_dotenv()

from src.app import create_app, run_interactive_chat, process_query
from src.utils.logger import setup_logger

logger = setup_logger()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="KubeSphere AI 助手")
    parser.add_argument("--interactive", "-i", action="store_true", help="启动交互式聊天模式")
    parser.add_argument("--api", "-a", action="store_true", help="启动API服务")
    parser.add_argument("--port", "-p", type=int, default=8000, help="API服务端口")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API服务主机")
    return parser.parse_args()

def test():
    result = process_query("什么是Kubernetes?", 1)
    print("\n回答:", result["answer"])
    result = process_query("它和Docker有什么关系?", 1)
    print("\n回答:", result["answer"])

def main():
    """主函数"""
    args = parse_args()
    
    if args.interactive:
        logger.info("启动交互式聊天模式")
        run_interactive_chat()
    elif args.api:
        logger.info(f"启动API服务 在 {args.host}:{args.port}")
        from src.api.endpoints import start_api_server
        start_api_server(host=args.host, port=args.port)
    else:
        # 如果没有提供任何参数，默认启动交互式模式
        logger.info("未提供参数，默认启动交互式聊天模式")
        
        # 运行测试
        test()

        run_interactive_chat()

if __name__ == "__main__":
    main()
