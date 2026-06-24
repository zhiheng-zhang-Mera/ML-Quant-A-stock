# llm_analyst.py
import json
import requests
from config import PipelineConfig

class LLMTextAnalyst:
    def __init__(self, api_key: str, config: PipelineConfig):
        self.api_key = api_key
        self.config = config
        
    def analyze_news_to_views(self, corpus: list, symbols: list) -> dict:
        """Parses corpus sentiment into structured alpha signal weights via multi-provider endpoint APIs."""
        if not self.api_key or "mock" in self.api_key:
            # Safe pipeline fallback operational mode inside mock testing sandboxes
            return {sym: 0.0005 for sym in symbols}
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = (
            f"Analyze the following financial context data and output numeric directional views for: {symbols}.\n"
            f"Context: {json.dumps(corpus)}\n"
            f"Respond STRICTLY with a valid JSON map where keys are asset symbols and values are float returns."
        )
        
        payload = {
            "model": self.config.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                f"{self.config.LLM_API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.LLM_TIMEOUT
            )
            res_json = response.json()
            content_str = res_json['choices'][0]['message']['content']
            parsed_views = json.loads(content_str)
            return {sym: float(parsed_views.get(sym, 0.0)) for sym in symbols}
        except Exception:
            # Resilient systemic degradation tracking default allocation structures
            return {sym: 0.0 for sym in symbols}