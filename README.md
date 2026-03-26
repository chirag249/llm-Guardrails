# 🛡️ SmartGuard - LLM Guardrails System

A machine learning-based guardrail system that classifies LLM prompts as safe or harmful in real-time. This project implements Track A (Pre-trained Model) using Google's Gemini as the core classification engine, backed by a high-speed caching layer.

## 📝 Project Creation & Setup

### Requirements
- Python 3.10+
- Gemini API key (free from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup Instructions (< 5 commands)

1. **Clone the repository and enter the directory:**
   ```bash
   cd llm_guard
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   ```bash
   cp .env.example .env
   # Open .env and insert your GEMINI_API_KEY
   ```

## 🚀 How to Run

**1. Run the Evaluation Pipeline:**
Executes the full 45-prompt red-team test suite, comparing the ML model against a baseline.
```bash
python src/evaluator.py
```

**2. Launch the Interactive Dashboard:**
Starts the Streamlit web interface for live testing, threshold sweeping, and failure analysis.
```bash
streamlit run src/dashboard.py
```

## 📁 Purpose of Each File

- **`src/classifier.py`**: The core classification engine. Uses a pre-trained ML model (Gemini) mixed with an in-memory caching mechanism to classify prompts across four categories (Jailbreak, PII, Toxic, Injection).
- **`src/evaluator.py`**: Runs the full red-team testing suite against the baseline and computes overall accuracy, precision, recall, and false positive rates.
- **`src/dashboard.py`**: An interactive Streamlit dashboard serving as the user interface for live prompt testing, viewing aggregate metrics, and analyzing threshold trade-offs.
- **`src/red_team_suite.py`**: Contains the curated dataset of 45 test prompts (30 attacks + 15 benign) with ground-truth safety labels.
- **`src/llm_backend.py`**: An integration example demonstrating how to wrap a live LLM generation function with the SmartGuard firewall.

## 🧠 Model Choice & Track Justification

**Track Chosen:** Track A (Pre-trained classifier model)

**Justification:** 
We selected a pre-trained ML model (Google Gemini) because simple keyword blocklists lack semantic understanding and easily fail against indirect phrasing, hypotheticals, and context-embedded attacks. Utilizing a pre-trained language model allows SmartGuard to deeply analyze the *intent* of the prompt rather than matching finite string patterns. To offset the latency associated with API-based models, we implemented a TTL caching layer that accelerates repeated or common queries.

## ⏱️ Latency & Performance Results

- **P95 Latency (Cached):** < 1ms (CPU-optimized memory cache)
- **P95 Latency (Cache Miss):** ~500-800ms (API dependent)

This architecture ensures that the system meets realistic performance constraints for recurring requests while maintaining the reasoning capabilities of ML for novel attacks.

- **Block Rate on Attacks:** ~80-85%
- **False Positive Rate:** < 15%
