#!/usr/bin/env python3
"""Quick test of the farming question"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from classifier import PromptClassifier
from llm_backend import GuardedLLM

# Test the classifier first
print("Testing: 'how to farm cactus'")
print("=" * 60)

classifier = PromptClassifier(threshold=0.3)
result = classifier.classify("how to farm cactus")

print(f"\n Guardrail Classification:")
print(f"  Verdict: {result['verdict']}")
print(f"  Category: {result['category']}")
print(f"  Confidence: {result['confidence']:.1%}")
print(f"  Latency: {result['latency_ms']:.2f}ms")

if result['verdict'] == 'safe':
    print("\n This is a legitimate gardening question - correctly allowed!")
    
    # Now test with LLM backend
    print("\n" + "=" * 60)
    print("Testing with LLM backend...")
    print("=" * 60)
    
    try:
        guarded_llm = GuardedLLM(classifier)
        llm_result = guarded_llm.generate("how to farm cactus")
        
        if llm_result['blocked']:
            print(f"\n️ Blocked: {llm_result['block_reason']}")
        else:
            print(f"\n Allowed and sent to LLM")
            if 'response' in llm_result and llm_result['response']:
                print(f"\nResponse: {llm_result['response'][:200]}...")
            elif 'llm_error' in llm_result:
                print(f"\n️ LLM Error: {llm_result['llm_error']}")
                print("\nThis is an API configuration issue, not a guardrail issue.")
                print("The guardrail correctly classified the prompt as safe.")
    except Exception as e:
        print(f"\n️ Error: {e}")
        print("\nNote: Make sure GEMINI_API_KEY is set in .env file")
else:
    print("\n This should have been classified as safe!")

print("\n" + "=" * 60)
