"""
FastAPI Portfolio Management Application with LangGraph AI Integration
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from typing import Dict, Any

# Import our LangGraph portfolio agent
from langgraph_model import portfolio_agent, chat_with_portfolio_agent
from config import get_model_info, update_model, get_gpt5_variants

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Management with LangGraph AI",
    description="A portfolio management application powered by LangGraph and OpenAI",
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main portfolio management interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: Dict[str, Any]):
    """
    Chat endpoint that integrates with the LangGraph portfolio agent.
    
    Expected request format:
    {
        "message": "user message here",
        "portfolio": {"AAPL": 30, "GOOGL": 25},
        "available_tags": ["asset_class", "region"],
        "tag_definitions": {"asset_class": ["equity", "fixed income"], "region": ["us", "non us"]}
    }
    """
    try:
        user_message = request.get("message", "")
        portfolio = request.get("portfolio", {})
        available_tags = request.get("available_tags", [])
        tag_definitions = request.get("tag_definitions", {})
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        print(f"🔍 Debug: Received message: {user_message}")
        print(f"🔍 Debug: Portfolio: {portfolio}")
        print(f"🔍 Debug: Available tags: {available_tags}")
        print(f"🔍 Debug: Tag definitions: {tag_definitions}")
        
        # Get response from LangGraph agent
        agent_response = chat_with_portfolio_agent(user_message, portfolio, available_tags, tag_definitions)
        
        print(f"🔍 Debug: LangGraph response: {agent_response['response']}")
        print(f"🔍 Debug: LangGraph portfolio state: {agent_response['portfolio_state']}")
        print(f"🔍 Debug: LangGraph changes: {agent_response['changes']}")
        
        return {
            "success": True,
            "response": agent_response['response'],
            "portfolio": agent_response['portfolio_state'],  # Return updated portfolio state
            "changes": agent_response['changes'],  # Return what changed
            "model_info": get_model_info()
        }
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/model-info")
async def get_model_information():
    """Get information about the current AI model configuration."""
    from config import get_model_info, get_gpt5_variants
    
    model_info = get_model_info()
    gpt5_info = get_gpt5_variants()
    
    return {
        **model_info,
        "gpt5_variants": gpt5_info
    }

@app.post("/api/update-model")
async def update_ai_model(request: Dict[str, str]):
    """
    Update the AI model being used.
    
    Expected request format:
    {
        "model": "gpt-4o"
    }
    """
    try:
        new_model = request.get("model", "")
        if not new_model:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        success = update_model(new_model)
        if success:
            return {
                "success": True,
                "message": f"Model updated to {new_model}",
                "model_info": get_model_info()
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Portfolio Management with LangGraph AI",
        "model_info": get_model_info()
    }

if __name__ == "__main__":
    print("🚀 Starting Portfolio Management Application with LangGraph AI")
    print("=" * 60)
    
    # Display model information
    model_info = get_model_info()
    gpt5_info = get_gpt5_variants()
    
    print(f"🤖 AI Model: {model_info['current_model']}")
    print(f"📝 Description: {model_info['description']}")
    print(f"⭐ Recommended: {model_info['recommended_model']}")
    print()
    
    print("🆕 GPT-5 Models Available:")
    for model, desc in gpt5_info['gpt5_variants'].items():
        print(f"   • {model}: {desc}")
    print()
    
    print(f"🔄 All Available Models: {', '.join(model_info['available_models'])}")
    print()
    
    if not get_model_info().get("current_model"):
        print("⚠️  WARNING: No OpenAI API key configured!")
        print("   Please set OPENAI_API_KEY in your environment or .env file")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        print("   Example .env file content:")
        print("   OPENAI_API_KEY=your_actual_api_key_here")
        print("   OPENAI_MODEL=gpt-5")
        print()
    
    print("🌐 Starting server at http://localhost:8000")
    print("📚 API documentation available at http://localhost:8000/docs")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
