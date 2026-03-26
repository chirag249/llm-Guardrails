"""
FastAPI server for SmartGuard
REST API for prompt classification
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from classifier import PromptClassifier

# Initialize FastAPI app
app = FastAPI(
    title="SmartGuard API",
    description="LLM Guardrails - Prompt Safety Classification API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = PromptClassifier(threshold=0.6)

# Request/Response models
class ClassifyRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to classify")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional threshold override")

class ClassifyResponse(BaseModel):
    verdict: str
    category: str
    confidence: float
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    threshold: float

class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)

# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "threshold": classifier.threshold
    }

@app.post("/classify", response_model=ClassifyResponse)
async def classify_prompt(request: ClassifyRequest):
    """
    Classify a prompt as safe or unsafe
    
    Returns verdict, category, confidence score, and latency
    """
    try:
        # Use custom threshold if provided
        if request.threshold is not None:
            original_threshold = classifier.threshold
            classifier.set_threshold(request.threshold)
            result = classifier.classify(request.prompt)
            classifier.set_threshold(original_threshold)
        else:
            result = classifier.classify(request.prompt)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/threshold")
async def update_threshold(update: ThresholdUpdate):
    """Update the global classification threshold"""
    try:
        classifier.set_threshold(update.threshold)
        return {
            "status": "success",
            "threshold": classifier.threshold
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/threshold")
async def get_threshold():
    """Get current threshold"""
    return {"threshold": classifier.threshold}

@app.post("/classify/batch")
async def classify_batch(prompts: list[str]):
    """Classify multiple prompts"""
    try:
        results = classifier.classify_batch(prompts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
