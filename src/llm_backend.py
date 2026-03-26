"""
LLM Backend Integration
Connects guardrails to Gemini API (Free Tier 3 Flash Preview)
"""

import os
from typing import Dict, Optional, List
import time
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMBackend:
    """
    Interface for Gemini API
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize Gemini backend
        
        Args:
            model: Model name. Defaults to gemini-3-flash-preview.
        """
        self.model = model or os.getenv("LLM_MODEL", "gemini-3-flash-preview")
        self.provider = "gemini"
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        
        # Initialize Gemini client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)
    
    def generate(self, prompt: str) -> Dict:
        """
        Generate response from Gemini
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict with response, latency, and metadata
        """
        start_time = time.perf_counter()
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            text = response.text
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "response": text,
                "latency_ms": round(latency_ms, 2),
                "provider": "gemini",
                "model": self.model
            }
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
            "error": str(e),
            "response": "LLM temporarily unavailable. Try again later.",
                "latency_ms": round(latency_ms, 2),
                "provider": "gemini",
                "model": self.model
            }


class GuardedLLM:
    """
    LLM with guardrails
    Classifies prompts before sending to LLM
    """
    
    def __init__(self, classifier, llm_backend: Optional[LLMBackend] = None):
        """
        Initialize guarded LLM
        
        Args:
            classifier: PromptClassifier instance
            llm_backend: LLMBackend instance (optional)
        """
        self.classifier = classifier
        self.llm_backend = llm_backend or LLMBackend()
        self.blocked_count = 0
        self.allowed_count = 0
    
    def generate(self, prompt: str, bypass_guardrails: bool = False) -> Dict:
        """
        Generate response with guardrails
        
        Args:
            prompt: User prompt
            bypass_guardrails: Skip safety check (for testing)
            
        Returns:
            Dict with classification, response (if allowed), and metadata
        """
        # Classify prompt
        classification = self.classifier.classify(prompt)
        
        result = {
            "prompt": prompt,
            "classification": classification,
            "blocked": False,
            "response": None,
            "llm_latency_ms": 0,
            "total_latency_ms": classification["latency_ms"]
        }
        
        # Check if blocked
        if classification["verdict"] == "unsafe" and not bypass_guardrails:
            self.blocked_count += 1
            result["blocked"] = True
            result["block_reason"] = f"Classified as {classification['category']} with {classification['confidence']:.0%} confidence"
            return result
        
        # Allow through to LLM
        self.allowed_count += 1
        llm_result = self.llm_backend.generate(prompt)
        
        result["response"] = llm_result.get("response")
        result["llm_latency_ms"] = llm_result.get("latency_ms", 0)
        result["total_latency_ms"] += llm_result.get("latency_ms", 0)
        result["llm_provider"] = "gemini"
        result["llm_model"] = llm_result.get("model")
        
        if "error" in llm_result:
            result["llm_error"] = llm_result["error"]
        
        return result
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        total = self.blocked_count + self.allowed_count
        return {
            "total_prompts": total,
            "blocked": self.blocked_count,
            "allowed": self.allowed_count,
            "block_rate": self.blocked_count / total if total > 0 else 0
        }


def test_llm_backend():
    """Test LLM backend connection"""
    print("Testing Gemini Flash Connection")
    print("=" * 60)
    
    try:
        backend = LLMBackend()
        result = backend.generate("Say 'Hello from Gemini Flash!' in one sentence.")
        
        if "error" in result:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Response: {result['response'][:100]}...")
            print(f"    Latency: {result['latency_ms']:.2f}ms")
        
    except Exception as e:
        print(f"    Failed: {str(e)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("SmartGuard Gemini Backend Test")
    print("=" * 60)
    
    if not os.getenv("GEMINI_API_KEY"):
        print("️  GEMINI_API_KEY not found in .env file")
    else:
        test_llm_backend()
