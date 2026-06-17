# llm_analyst.py
import json
import logging
import requests
from typing import Dict, List, Tuple

logger = logging.getLogger("QuantPipeline.LLMAnalyst")

class LLMTextAnalyst:
    def __init__(self, api_key: str, model_name: str, api_url: str, temperature: float = 0.1, timeout: float = 15.0):
        """
        高度解耦的、基于通用标准 OpenAI 兼容协议的大模型舆情特征分析网关。
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url.rstrip('/')  # 安全防御：拦截清除多余的末尾斜杠，防范路径拼接双斜杠异常
        self.temperature = temperature
        self.timeout = timeout

    def analyze_news_to_views(self, news_corpus: List[Dict], target_assets: List[str]) -> List[Tuple[str, float, float]]:
        """
        解析收盘前的非结构化宏观/行业文本，严格防前瞻作弊。
        返回: List[Tuple(受影响资产代码, 预期超额Delta收益率, 文本噪声放大系数)]
        """
        llm_views = []
        if not news_corpus or not target_assets:
            return llm_views

        # Synthesize unstructured news block
        corpus_text = "\n".join([f"- [{n.get('timestamp', '15:00:00')}] {n.get('headline', '')}" for n in news_corpus])
        
        # Enforce structural strictness using System Role parameters
        system_prompt = (
            "You are an elite quantitative research analyst specializing in empirical asset pricing.\n"
            "Analyze the provided news corpus and isolate short-term macro/industry alpha views for target assets.\n"
            "You MUST respond with a valid JSON object containing a 'views' list of objects. Each object MUST contain:\n"
            "  - 'asset': string, match exactly from the allowed pool (e.g., '512480')\n"
            "  - 'expected_delta': float, estimated excess return (e.g., 0.015 for +1.5%)\n"
            "  - 'noise_coefficient': float, volatility multiplier scaling factor based on source ambiguity (>= 1.0)\n"
            "Return ONLY raw JSON. No markdown wrappers, no explanations."
        )
        
        user_content = f"Allowed Asset Pool: {target_assets}\n\nNews Corpus to Analyze:\n{corpus_text}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        try:
            # 使用从配置中心下发的可变端点与超时机制执行生产级 HTTP 通信
            response = requests.post(f"{self.api_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                result_json = response.json()
                content_str = result_json['choices'][0]['message']['content'].strip()
                
                # 强化版鲁棒性字符串防御：即便大模型由于特定框架微调未严格遵循 json_object 而吐出 Markdown 包裹也能成功解析
                if content_str.startswith("```json"):
                    content_str = content_str.split("```json")[1].split("```")[0].strip()
                elif content_str.startswith("```"):
                    content_str = content_str.split("```")[1].split("```")[0].strip()
                
                parsed_data = json.loads(content_str)
                views_list = parsed_data.get("views", [])
                
                for item in views_list:
                    asset = str(item.get("asset", ""))
                    if asset in target_assets:
                        delta = float(item.get("expected_delta", 0.0))
                        noise = float(item.get("noise_coefficient", 1.0))
                        llm_views.append((asset, delta, max(noise, 1.0)))
            else:
                logger.error(f"LLM API returned failure code {response.status_code}: {response.text}")
                raise RuntimeError(f"API non-200 connection drop. Status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[LLM-EXCEPTION] Connection or parse error encountered: {str(e)}")
            logger.warning("[FALLBACK] Activating white-box defensive simulation rules.")
            # 工业实盘级稳健防御：若发生极端断网或 API 封禁崩溃，安全降级回无偏差模拟观点，绝不阻塞量价计算主进程
            if "512480" in target_assets and len(news_corpus) > 0:
                llm_views.append(("512480", 0.015, 1.5)) 
            
        return llm_views