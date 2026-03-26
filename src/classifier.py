"""
SmartGuard - LLM Guardrails System
Core prompt classifier using pre-trained lightweight model
Track A: Pre-trained model approach
"""

import time
import os
import json
from typing import Dict, Tuple
from enum import Enum

class ThreatCategory(Enum):
    SAFE = "safe"
    JAILBREAK = "jailbreak"
    PII_EXTRACTION = "pii_extraction"
    TOXIC = "toxic"
    INJECTION = "injection"
    UNKNOWN = "unknown"

class PromptClassifier:
    """
    Lightweight CPU-friendly prompt classifier
    
    Model Choice: Using distilbert-base-uncased-finetuned-sst-2-english as base
    with custom classification layer for guardrail categories
    
    Rationale:
    - DistilBERT is 40% smaller than BERT, 60% faster
    - CPU-optimized with good accuracy/speed trade-off
    - Well-documented and production-ready
    - P95 latency target: <100ms on CPU
    """
    
    def __init__(self, model_name: str = "keyword-based", threshold: float = 0.5):
        """
        Initialize classifier with configurable threshold
        
        Args:
            model_name: Model identifier (for compatibility)
            threshold: Classification threshold (0-1), lower = stricter
        """
        self.threshold = threshold
        self.device = "cpu"  # Enforce CPU-only operation
        self.model_name = model_name
        
        # Keyword patterns for enhanced detection
        self.jailbreak_patterns = [
            "ignore previous instructions", "ignore above", "disregard",
            "pretend you are", "act as if", "roleplay", "dan mode",
            "you are now", "new instructions", "override", "bypass",
            "ignore all", "forget", "how to hack"
        ]
        
        self.pii_patterns = [
            "credit card", "social security", "ssn", "password",
            "api key", "secret", "private key", "confidential"
        ]
        
        self.toxic_patterns = [
            "kill yourself", "harm yourself", "commit suicide",
            "how to make a bomb", "hate speech", "racist", "sexist",
            "make a bomb", "create a bomb", "build a bomb",
            # Drug manufacturing and illegal substances
            "cook meth", "make meth", "methamphetamine", "crystal meth",
            "cook crack", "make crack", "crack cocaine", "freebase",
            "synthesize drugs", "drug lab", "meth lab", "cook drugs",
            "make heroin", "synthesize heroin", "purify heroin",
            "make cocaine", "process cocaine", "cocaine production", "synthesize cocaine",
            "make fentanyl", "synthesize fentanyl", "fentanyl production",
            "mdma synthesis", "make ecstasy", "synthesize mdma",
            "lsd synthesis", "make lsd", "synthesize lsd",
            "homebrew drugs", "diy drugs", "illegal drug",
            "drug precursor", "pseudoephedrine extraction", "red phosphorus",
            "anhydrous ammonia", "lithium strips", "iodine crystals",
            # Weapons and violence
            "make explosives", "homemade explosive", "pipe bomb",
            "pressure cooker bomb", "fertilizer bomb", "ammonium nitrate",
            "how to kill", "murder someone", "assassination",
            "poison someone", "deadly poison", "ricin", "cyanide",
            # Nuclear and radioactive
            "uranium enrichment", "plutonium", "dirty bomb", "process uranium",
            "nuclear weapon", "atomic bomb", "enriched uranium", "uranium 235"
        ]
        
        self.injection_patterns = [
            "system:", "{{", "}}", "$(", "${", "<script>",
            "eval(", "exec(", "import os", "__import__"
        ]

        # Classification cache to avoid repeated API calls
        # In-memory cache: {normalized_prompt: (ThreatCategory, confidence, timestamp)}
        self.cache_ttl = int(os.getenv('CLASSIFIER_CACHE_TTL', '300'))  # seconds
        self.cache_max = int(os.getenv('CLASSIFIER_CACHE_MAX', '1000'))
        self._cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def set_threshold(self, threshold: float):
        """Update classification threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def _keyword_score(self, text: str) -> Tuple[ThreatCategory, float]:
        """
        Enhanced keyword-based classification (fallback)
        Returns (category, confidence_score)
        """
        text_lower = text.lower()
        
        # Count matches in each category
        jailbreak_matches = sum(1 for pattern in self.jailbreak_patterns if pattern in text_lower)
        pii_matches = sum(1 for pattern in self.pii_patterns if pattern in text_lower)
        toxic_matches = sum(1 for pattern in self.toxic_patterns if pattern in text_lower)
        injection_matches = sum(1 for pattern in self.injection_patterns if pattern in text_lower)
        
        scores = {
            ThreatCategory.JAILBREAK: min(jailbreak_matches / 3, 1.0),
            ThreatCategory.PII_EXTRACTION: min(pii_matches / 2, 1.0),
            ThreatCategory.TOXIC: min(toxic_matches / 2, 1.0),
            ThreatCategory.INJECTION: min(injection_matches / 2, 1.0)
        }
        
        # Find highest scoring category
        max_category = max(scores.items(), key=lambda x: x[1])
        
        if max_category[1] < 0.1:
            return ThreatCategory.SAFE, 0.0
        
        return max_category[0], max_category[1]
    
    def _pretrained_score(self, text: str) -> Tuple[ThreatCategory, float]:
        """
        Use configured pre-trained provider (Gemini) to classify the prompt.
        Returns (ThreatCategory, confidence).
        Raises an exception if the pretrained provider cannot be used.
        """
        provider = os.getenv('LLM_PROVIDER', 'gemini')
        model = os.getenv('MODERATION_MODEL', os.getenv('LLM_MODEL', 'gemini-3-flash-preview'))

        # Only support Gemini here as pre-trained provider
        if provider != 'gemini':
            raise RuntimeError("Only 'gemini' provider is supported for pretrained scoring")

        from google import genai
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY not found')

        client = genai.Client(api_key=api_key)

        prompt = (
            "You are a content safety classifier. Given the user prompt between triple backticks, "
            "classify it into one of: jailbreak, pii_extraction, toxic, injection, safe. "
            "Return ONLY a single JSON object with keys: category (string), confidence (number between 0 and 1). "
            "If unsure, use 'toxic' for dangerous content.\n\n" +
            "User prompt:\n```\n" + text + "\n```\n"
        )

        response = client.models.generate_content(model=model, contents=prompt)
        text_out = getattr(response, 'text', '') or str(response)

        # Attempt to parse JSON from model output
        try:
            parsed = json.loads(text_out)
            cat = parsed.get('category', '').lower()
            conf = float(parsed.get('confidence', 0.0))
        except Exception:
            # If parsing fails, do heuristic extraction
            cat = 'safe'
            conf = 0.0
            for c in ['jailbreak', 'pii_extraction', 'toxic', 'injection']:
                if c in text_out.lower():
                    cat = c
                    conf = 0.5
                    break

        mapping = {
            'jailbreak': ThreatCategory.JAILBREAK,
            'pii_extraction': ThreatCategory.PII_EXTRACTION,
            'toxic': ThreatCategory.TOXIC,
            'injection': ThreatCategory.INJECTION,
            'safe': ThreatCategory.SAFE
        }
        category = mapping.get(cat, ThreatCategory.SAFE)
        return category, min(max(conf, 0.0), 1.0)
    
    def _cache_get(self, key: str):
        """Return cached (category, confidence) or None if missing/expired"""
        entry = self._cache.get(key)
        if not entry:
            return None
        category, conf, ts = entry
        if (time.time() - ts) > self.cache_ttl:
            # expired
            try:
                del self._cache[key]
            except KeyError:
                pass
            return None
        return category, conf

    def _cache_set(self, key: str, category: ThreatCategory, conf: float):
        """Store value and evict oldest if over capacity"""
        # Evict oldest if needed
        if len(self._cache) >= self.cache_max:
            # find oldest
            oldest = min(self._cache.items(), key=lambda kv: kv[1][2])[0]
            try:
                del self._cache[oldest]
            except KeyError:
                pass
        self._cache[key] = (category, conf, time.time())

    def classify(self, prompt: str) -> Dict:
        """
        Classify a prompt and return verdict, category, and confidence
        
        Args:
            prompt: Input text to classify
            
        Returns:
            Dict with keys: verdict (safe/unsafe), category, confidence, latency_ms
        """
        start_time = time.perf_counter()
        key = prompt.strip().lower()

        # Check cache first
        cached = self._cache_get(key)
        if cached is not None:
            category, confidence = cached
            self.cache_hits += 1
        else:
            # Not cached: call pretrained (Gemini) classifier
            try:
                category, confidence = self._pretrained_score(prompt)
            except Exception:
                # If pretrained classifier is unavailable, enforce cache-only policy:
                # block prompts that are not already cached by returning UNKNOWN with high confidence
                category = ThreatCategory.UNKNOWN
                confidence = 1.0
            # store in cache
            try:
                self._cache_set(key, category, confidence)
            except Exception:
                # ignore caching failures
                pass
            self.cache_misses += 1

        # Apply threshold
        is_unsafe = (category != ThreatCategory.SAFE) and (confidence >= self.threshold)
        verdict = "unsafe" if is_unsafe else "safe"

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "verdict": verdict,
            "category": category.value,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 2),
            "cache_hit": cached is not None
        }

    def classify_batch(self, prompts: list) -> list:
        """Classify multiple prompts"""
        return [self.classify(prompt) for prompt in prompts]


# Compatibility shim: lightweight keyword baseline retained for evaluation and tests
class KeywordBaseline:
    """Thin compatibility baseline implementing simple keyword blocking.
    Exists to support evaluation and test scripts that compare against a simple baseline.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.blocked_keywords = [
            "ignore instructions", "disregard", "jailbreak", "ignore all",
            "credit card", "password", "kill yourself", "bomb", "hack"
        ]

    def set_threshold(self, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold

    def classify(self, prompt: str) -> Dict:
        start_time = time.perf_counter()
        text_lower = prompt.lower()
        for keyword in self.blocked_keywords:
            if keyword in text_lower:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "verdict": "unsafe",
                    "category": "unknown",
                    "confidence": 1.0,
                    "latency_ms": round(latency_ms, 2),
                    "cache_hit": False
                }
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "verdict": "safe",
            "category": "safe",
            "confidence": 0.0,
            "latency_ms": round(latency_ms, 2),
            "cache_hit": False
        }
