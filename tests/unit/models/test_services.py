from typing import Dict, Any, Optional, List
import anthropic
import openai



class LLMService:
    """大语言模型服务，支持多种LLM提供商"""
    
    def __init__(
        self, 
        model_name: str = "qwen-plus",
        anthropic_api_key: Optional[str] = '',
        openai_api_key: Optional[str] = '',
        qwen_api_key: Optional[str] = '',
        glm_api_key: Optional[str] = '',
        temperature: float = 0.2,
        base_url: Optional[str] = None,
        max_tokens: int = 4000
    ):
        """
        初始化LLM客户端
        
        Args:
            provider: 模型提供商，支持"anthropic", "openai", "qwen", "glm"
            model_name: 模型名称
            anthropic_api_key: Anthropic API密钥
            openai_api_key: OpenAI API密钥
            qwen_api_key: 阿里Qwen API密钥
            glm_api_key: 智谱GLM API密钥
            base_url: 自定义API基础URL，用于本地API服务或代理
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # 确定提供商
        self.provider = "qwen"
        # 初始化客户端
        if self.provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("使用Claude模型需要提供Anthropic API密钥")
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        elif self.provider == "openai":
            if not openai_api_key:
                raise ValueError("使用OpenAI模型需要提供OpenAI API密钥")
            client_args = {"api_key": openai_api_key}
            if self.base_url:
                client_args["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_args)
        elif self.provider == "qwen":
            if not qwen_api_key:
                raise ValueError("使用Qwen模型需要提供Qwen API密钥")
            client_args = {"api_key": qwen_api_key}
            # Qwen API通常使用OpenAI兼容接口
            if not self.base_url:
                self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            client_args["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_args)
        elif self.provider == "glm":
            if not glm_api_key:
                raise ValueError("使用GLM模型需要提供GLM API密钥")
            client_args = {"api_key": glm_api_key}
            # GLM API通常使用OpenAI兼容接口
            if not self.base_url:
                self.base_url = "https://open.bigmodel.cn/api/paas/v4"
            client_args["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_args)
        else:
            raise ValueError(f"不支持的模型提供商: {self.provider}")
        
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 **kwargs) -> Dict[str, Any]:
        """
        生成文本响应
        
        Args:
            prompt: 用户输入提示
            system_prompt: 系统提示
            max_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            **kwargs: 其他参数
            
        Returns:
            包含生成文本和元数据的字典
        """
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, max_tokens, temperature, **kwargs)
        elif self.provider in ["openai", "qwen", "glm"]:
            return self._generate_openai_compatible(prompt, system_prompt, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"不支持的模型提供商: {self.provider}")
    
    def _generate_anthropic(self, prompt, system_prompt, max_tokens, temperature, **kwargs):
        """使用Anthropic API生成文本"""
        messages = [{"role": "user", "content": prompt}]
        
        # 添加系统提示
        system = system_prompt or "你是一个有用的AI助手。"
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return {
            "text": response.content[0].text,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    
    def _generate_openai_compatible(self, prompt, system_prompt, max_tokens, temperature, **kwargs):
        """使用OpenAI兼容API生成文本（适用于OpenAI、Qwen和GLM）"""
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # 针对不同提供商
            if self.provider == "qwen":
                if "model" not in kwargs:
                    kwargs["model"] = self.model_name
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            elif self.provider == "glm":
                if "model" not in kwargs:
                    kwargs["model"] = self.model_name
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            
            return {
                "text": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
            }
        except Exception as e:
            logger.error(f"{self.provider} API调用失败: {str(e)}")
            # 重新抛出异常，但添加更多上下文
            raise RuntimeError(f"{self.provider} API调用失败: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # # 使用Claude API
    # try:
    #     claude_client = LLMService(
    #         provider="anthropic", 
    #         model_name="claude-3-sonnet-20240229"
    #     )
    #     response = claude_client.generate(
    #         prompt="请简要介绍一下Python的优势",
    #         system_prompt="你是一个专业的编程教师，擅长简洁明了地解释技术概念。"
    #     )
    #     print(f"Claude响应: {response['text']}")
    #     print(f"使用模型: {response['model']}")
    #     print(f"Token使用: {response['usage']}")
    # except Exception as e:
    #     print(f"Claude API调用失败: {str(e)}")
    
    # 使用Qwen API
    try:
        qwen_client = LLMService(
            # provider="qwen",
            model_name="qwen-plus",  # 根据实际可用的模型名称调整
            qwen_api_key="sk-c1c95e661b1443f78f10c86fe570585e"  # 或设置环境变量QWEN_API_KEY
        )
        response = qwen_client.generate(
            prompt="请简要介绍一下Python的优势",
            system_prompt="你是一个专业的编程教师，擅长简洁明了地解释技术概念。"
        )
        print(f"Qwen响应: {response['text']}")
    except Exception as e:
        print(f"Qwen API调用失败: {str(e)}")
    
    # # 使用GLM API
    # try:
    #     glm_client = LLMService(
    #         provider="glm",
    #         model_name="glm-4",  # 根据实际可用的模型名称调整
    #         glm_api_key="your_glm_api_key"  # 或设置环境变量GLM_API_KEY
    #     )
    #     response = glm_client.generate(
    #         prompt="请简要介绍一下Python的优势",
    #         system_prompt="你是一个专业的编程教师，擅长简洁明了地解释技术概念。"
    #     )
    #     print(f"GLM响应: {response['text']}")
    # except Exception as e:
    #     print(f"GLM API调用失败: {str(e)}")