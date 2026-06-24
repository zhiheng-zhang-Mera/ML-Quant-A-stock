# llm_analyst.py
import json
import requests
import logging
from config import PipelineConfig

logger = logging.getLogger("QuantPipeline.LLM")

class LLMTextAnalyst:
    def __init__(self, api_key: str, config: PipelineConfig):
        self.api_key = api_key
        self.config = config
        
    def analyze_news_to_views(self, corpus: list, symbols: list) -> dict:
        """Parses corpus sentiment into structured alpha signal weights via multi-provider endpoint APIs."""
        provider = self.config.LLM_PROVIDER.upper()
        
        # 1. 动态路由网络拓扑与鉴权护栏
        if provider == "OLLAMA":
            api_url = self.config.OLLAMA_API_URL
            model_name = self.config.OLLAMA_MODEL_NAME
            headers = {"Content-Type": "application/json"}
        else:  # DEFAULT: DEEPSEEK
            api_url = self.config.DEEPSEEK_API_URL
            model_name = self.config.DEEPSEEK_MODEL_NAME
            if not self.api_key or "mock" in self.api_key:
                return {sym: 0.0005 for sym in symbols}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        
        prompt = (
            f"Analyze the following financial context data and output numeric directional views for: {symbols}.\n"
            f"Context: {json.dumps(corpus)}\n"
            f"Respond STRICTLY with a valid JSON map where keys are asset symbols and values are float returns. "
            f"Do not wrap your output in conversational markdown blocks outside the raw JSON object structure."
        )
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.LLM_TIMEOUT
            )
            res_json = response.json()
            content_str = res_json['choices'][0]['message']['content'].strip()
            
            # 🚨【工业级抗干扰防线】：部分本地小模型会强行夹带 ```json 标记，在此进行强制剥壳
            if content_str.startswith("```json"):
                content_str = content_str.split("```json")[1].split("```")[0].strip()
            elif content_str.startswith("```"):
                content_str = content_str.split("```")[1].split("```")[0].strip()
                
            parsed_views = json.loads(content_str)
            return {sym: float(parsed_views.get(sym, 0.0)) for sym in symbols}
        except Exception as e:
            # 本地模型崩溃或硬件响应不及时时的柔性降级保障机制
            logger.error(f"[LLM_ERROR] Generation aborted under provider {provider}. Error: {str(e)}")
            return {sym: 0.0 for sym in symbols}