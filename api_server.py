#!/usr/bin/env python3
"""
ReasonIt API Server

FastAPI-based REST API for the ReasonIt reasoning architecture.
Provides HTTP endpoints for reasoning, planning, and system management.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from reasonit import ReasonItApp, get_app
from models import ReasoningStrategy, ContextVariant, OutcomeType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global app instance
reasonit_app: Optional[ReasonItApp] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global reasonit_app
    
    # Startup
    logger.info("Starting ReasonIt API server...")
    try:
        reasonit_app = await get_app()
        logger.info("ReasonIt application initialized")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize ReasonIt: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ReasonIt API server...")
        if reasonit_app:
            await reasonit_app.close()


# Create FastAPI app
app = FastAPI(
    title="ReasonIt API",
    description="Advanced LLM reasoning architecture API",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ReasoningRequest(BaseModel):
    """Request model for reasoning endpoints."""
    
    query: str = Field(..., description="The question or problem to reason about")
    strategy: Optional[str] = Field(None, description="Reasoning strategy (cot, tot, mcts, self_ask, reflexion)")
    context_variant: str = Field("standard", description="Context variant (minified, standard, enriched, symbolic, exemplar)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_cost: float = Field(0.10, gt=0.0, description="Maximum cost per query")
    use_tools: bool = Field(True, description="Whether to use available tools")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class ReasoningResponse(BaseModel):
    """Response model for reasoning endpoints."""
    
    final_answer: str
    confidence_score: float
    total_cost: float
    total_time: float
    strategies_used: List[str]
    outcome: str
    reasoning_trace: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    timestamp: datetime


class PlanningRequest(BaseModel):
    """Request model for planning endpoints."""
    
    query: str = Field(..., description="The complex task to plan")
    decomposition_strategy: Optional[str] = Field(None, description="Decomposition strategy")


class PlanningResponse(BaseModel):
    """Response model for planning endpoints."""
    
    plan_id: str
    name: str
    description: str
    total_tasks: int
    estimated_cost: float
    estimated_time: float
    success_rate: float
    created_at: datetime


class SystemStatus(BaseModel):
    """System status response model."""
    
    status: str
    initialized: bool
    timestamp: float
    agents: Dict[str, str]
    controllers: Dict[str, str]
    tools: Dict[str, str]
    planning: Dict[str, str]
    memory: str
    session_manager: str


# Dependency to get ReasonIt app
async def get_reasonit_app() -> ReasonItApp:
    """Dependency to get the ReasonIt application instance."""
    if not reasonit_app:
        raise HTTPException(status_code=503, detail="ReasonIt application not initialized")
    return reasonit_app


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}


# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status(app: ReasonItApp = Depends(get_reasonit_app)):
    """Get comprehensive system status."""
    try:
        status = await app.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main reasoning endpoint
@app.post("/reason", response_model=ReasoningResponse)
async def reason(
    request: ReasoningRequest,
    background_tasks: BackgroundTasks,
    app: ReasonItApp = Depends(get_reasonit_app)
):
    """Execute reasoning for a query."""
    
    try:
        # Convert string enums to proper types
        strategy_enum = None
        if request.strategy:
            strategy_map = {
                "cot": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "tot": ReasoningStrategy.TREE_OF_THOUGHTS,
                "mcts": ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
                "self_ask": ReasoningStrategy.SELF_ASK,
                "reflexion": ReasoningStrategy.REFLEXION
            }
            strategy_enum = strategy_map.get(request.strategy.lower())
            if not strategy_enum and request.strategy:
                raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
        
        context_variant_map = {
            "minified": ContextVariant.MINIFIED,
            "standard": ContextVariant.STANDARD,
            "enriched": ContextVariant.ENRICHED,
            "symbolic": ContextVariant.SYMBOLIC,
            "exemplar": ContextVariant.EXEMPLAR
        }
        context_variant = context_variant_map.get(request.context_variant, ContextVariant.STANDARD)
        
        # Execute reasoning
        result = await app.reason(
            query=request.query,
            strategy=strategy_enum,
            context_variant=context_variant,
            confidence_threshold=request.confidence_threshold,
            max_cost=request.max_cost,
            use_tools=request.use_tools,
            session_id=request.session_id
        )
        
        # Convert reasoning trace to serializable format
        reasoning_trace = None
        if hasattr(result, 'reasoning_trace') and result.reasoning_trace:
            reasoning_trace = []
            for step in result.reasoning_trace:
                step_dict = {
                    "step_number": getattr(step, 'step_number', 0),
                    "content": getattr(step, 'content', ''),
                    "confidence": getattr(step, 'confidence', 0.0),
                    "cost": getattr(step, 'cost', 0.0)
                }
                reasoning_trace.append(step_dict)
        
        return ReasoningResponse(
            final_answer=result.final_answer,
            confidence_score=result.confidence_score,
            total_cost=result.total_cost,
            total_time=result.total_time,
            strategies_used=[s.value for s in result.strategies_used],
            outcome=result.outcome.value,
            reasoning_trace=reasoning_trace,
            session_id=getattr(result.request, 'session_id', None),
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Planning endpoint
@app.post("/plan", response_model=PlanningResponse)
async def create_plan(
    request: PlanningRequest,
    app: ReasonItApp = Depends(get_reasonit_app)
):
    """Create an execution plan for a complex task."""
    
    try:
        plan = await app.create_plan(
            query=request.query,
            decomposition_strategy=request.decomposition_strategy
        )
        
        return PlanningResponse(
            plan_id=plan.id,
            name=plan.name,
            description=plan.description,
            total_tasks=len(plan.tasks),
            estimated_cost=plan.total_cost,
            estimated_time=plan.total_time,
            success_rate=plan.success_rate,
            created_at=plan.created_at
        )
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quick reasoning endpoint with minimal parameters
@app.get("/quick")
async def quick_reason(
    q: str,
    strategy: Optional[str] = None,
    app: ReasonItApp = Depends(get_reasonit_app)
):
    """Quick reasoning endpoint with query parameter."""
    
    request = ReasoningRequest(
        query=q,
        strategy=strategy,
        confidence_threshold=0.7,
        max_cost=0.05
    )
    
    return await reason(request, BackgroundTasks(), app)


# Batch reasoning endpoint
@app.post("/batch")
async def batch_reason(
    requests: List[ReasoningRequest],
    app: ReasonItApp = Depends(get_reasonit_app)
):
    """Process multiple reasoning requests in batch."""
    
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 10 requests")
    
    results = []
    for req in requests:
        try:
            response = await reason(req, BackgroundTasks(), app)
            results.append({"success": True, "result": response})
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    
    return {"results": results}


# Available strategies endpoint
@app.get("/strategies")
async def get_strategies():
    """Get available reasoning strategies."""
    return {
        "strategies": [
            {
                "name": "cot",
                "full_name": "Chain of Thought",
                "description": "Sequential reasoning, good for math problems",
                "speed": "fast",
                "quality": "good"
            },
            {
                "name": "tot",
                "full_name": "Tree of Thoughts",
                "description": "Complex analysis with multiple solution paths",
                "speed": "medium",
                "quality": "high"
            },
            {
                "name": "mcts",
                "full_name": "Monte Carlo Tree Search",
                "description": "Exploration-based reasoning for creative problems",
                "speed": "slow",
                "quality": "excellent"
            },
            {
                "name": "self_ask",
                "full_name": "Self-Ask",
                "description": "Question decomposition for research tasks",
                "speed": "medium",
                "quality": "high"
            },
            {
                "name": "reflexion",
                "full_name": "Reflexion",
                "description": "Iterative improvement with learning from mistakes",
                "speed": "slow",
                "quality": "excellent"
            }
        ]
    }


# Metrics endpoint
@app.get("/metrics")
async def get_metrics(app: ReasonItApp = Depends(get_reasonit_app)):
    """Get system metrics and statistics."""
    try:
        status = await app.get_system_status()
        
        # Extract metrics from status
        metrics = {
            "system_status": status.get("initialized", False),
            "timestamp": status.get("timestamp", 0),
            "cost_summary": status.get("cost_summary", {}),
            "confidence_metrics": status.get("confidence_metrics", {}),
            "planning_metrics": status.get("planning_metrics", {})
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )


def main():
    """Run the API server."""
    port = int(os.getenv("REASONIT_API_PORT", "8000"))
    host = os.getenv("REASONIT_API_HOST", "0.0.0.0")
    reload = os.getenv("REASONIT_API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting ReasonIt API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()