"""
LangGraph Portfolio Management Agent

This module implements a LangGraph-based AI agent that can manage portfolios
and interact with users through natural language. It integrates with OpenAI's
latest models and provides tools for portfolio operations using proper LangGraph workflow.
"""

import os
import json
import re
from typing import Dict, Any, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state structure for LangGraph
class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    portfolio: Dict[str, Any]
    current_tool_results: List[Dict[str, Any]]  # Track tool execution results
    needs_final_response: bool  # Flag for when to generate final response

# Portfolio Management Tools
@tool
def add_ticker_to_portfolio(ticker: str, weight: float, portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
    """Add a new ticker to the portfolio with specified weight percentage."""
    ticker = ticker.upper()
    
    # Use provided portfolio or default to empty
    if portfolio is None:
        portfolio = {}
    
    # Convert portfolio to internal format if needed
    if isinstance(portfolio, dict) and "tickers" in portfolio:
        # Frontend format - convert to internal
        internal_portfolio = {}
        for ticker_data in portfolio["tickers"]:
            symbol = ticker_data["symbol"]
            # Preserve the ENTIRE ticker_data structure including tags
            if isinstance(ticker_data, dict) and len(ticker_data) > 2:
                # Has tags - preserve the complete structure
                internal_portfolio[symbol] = ticker_data
            else:
                # No tags - just store weight
                internal_portfolio[symbol] = ticker_data["weight"]
    else:
        # Already internal format
        internal_portfolio = portfolio.copy()
    
    # Check if ticker already exists
    if ticker in internal_portfolio:
        return {
            "success": False,
            "message": f"Ticker {ticker} already exists in portfolio",
            "portfolio": internal_portfolio,
            "action": "add_ticker",
            "ticker": ticker,
            "weight": weight
        }
    
    # Check total weight constraint - ensure all weights are floats
    total_weight = 0
    for ticker_data in internal_portfolio.values():
        if isinstance(ticker_data, dict):
            weight_value = ticker_data["weight"]
        else:
            weight_value = ticker_data
        
        # Convert to float if it's a string
        if isinstance(weight_value, str):
            weight_value = float(weight_value)
        total_weight += weight_value
    
    total_weight += weight
    if total_weight > 100:
        return {
            "success": False,
            "message": f"Adding {ticker} with {weight}% would exceed 100% total weight",
            "portfolio": internal_portfolio,
            "action": "add_ticker",
            "ticker": ticker,
            "weight": weight
        }
    
    # Add ticker
    internal_portfolio[ticker] = weight
    
    return {
        "success": True,
        "message": f"Added {ticker} with {weight}% weight",
        "portfolio": internal_portfolio,
        "action": "add_ticker",
        "ticker": ticker,
        "weight": weight
    }

@tool
def remove_ticker_from_portfolio(ticker: str, portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
    """Remove a ticker from the portfolio."""
    ticker = ticker.upper()
    
    # Use provided portfolio or default to empty
    if portfolio is None:
        portfolio = {}
    
    # Convert portfolio to internal format if needed
    if isinstance(portfolio, dict) and "tickers" in portfolio:
        # Frontend format - convert to internal
        internal_portfolio = {}
        for ticker_data in portfolio["tickers"]:
            symbol = ticker_data["symbol"]
            # Preserve the ENTIRE ticker_data structure including tags
            if isinstance(ticker_data, dict) and len(ticker_data) > 2:
                # Has tags - preserve the complete structure
                internal_portfolio[symbol] = ticker_data
            else:
                # No tags - just store weight
                internal_portfolio[symbol] = ticker_data["weight"]
    else:
        # Already internal format
        internal_portfolio = portfolio.copy()
    
    # Check if ticker exists
    if ticker not in internal_portfolio:
        return {
            "success": False,
            "message": f"Ticker {ticker} not found in portfolio",
            "portfolio": internal_portfolio,
            "action": "remove_ticker",
            "ticker": ticker
        }
    
    # Remove ticker
    removed_weight = internal_portfolio.pop(ticker)
    
    return {
        "success": True,
        "message": f"Removed {ticker} (was {removed_weight}%)",
        "portfolio": internal_portfolio,
        "action": "remove_ticker",
        "ticker": ticker
    }

@tool
def modify_ticker_weight(ticker: str, new_weight: float, portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
    """Modify the weight of an existing ticker in the portfolio."""
    ticker = ticker.upper()
    
    # Use provided portfolio or default to empty
    if portfolio is None:
        portfolio = {}
    
    # Convert portfolio to internal format if needed
    if isinstance(portfolio, dict) and "tickers" in portfolio:
        # Frontend format - convert to internal
        internal_portfolio = {}
        for ticker_data in portfolio["tickers"]:
            symbol = ticker_data["symbol"]
            ticker_weight = float(ticker_data["weight"])
            # Preserve existing tags if they exist
            if isinstance(ticker_data, dict) and len(ticker_data) > 2:
                # Has additional properties (tags) - preserve the entire structure
                internal_portfolio[symbol] = ticker_data
                print(f"🔍 Debug: Preserved tags for {symbol}: {ticker_data}")
            else:
                internal_portfolio[symbol] = ticker_weight
    else:
        # Already internal format
        internal_portfolio = portfolio.copy()
    
    print(f"🔍 Debug: Internal portfolio for modification: {internal_portfolio}")
    
    # Check if ticker exists
    if ticker not in internal_portfolio:
        return {
            "success": False,
            "message": f"Ticker {ticker} not found in portfolio",
            "portfolio": internal_portfolio,
            "action": "modify_weight",
            "ticker": ticker,
            "new_weight": new_weight
        }
    
    # Get current weight and check total weight constraint
    current_weight = internal_portfolio[ticker]
    if isinstance(current_weight, dict):
        current_weight = current_weight["weight"]
    
    # Calculate new total weight - ensure all weights are floats
    total_weight = 0
    for ticker_data in internal_portfolio.values():
        if isinstance(ticker_data, dict):
            weight = ticker_data["weight"]
        else:
            weight = ticker_data
        
        # Convert to float if it's a string
        if isinstance(weight, str):
            weight = float(weight)
        total_weight += weight
    
    # Convert current_weight to float if it's a string
    if isinstance(current_weight, str):
        current_weight = float(current_weight)
    
    # Calculate new total weight
    total_weight = total_weight - current_weight + new_weight
    
    if total_weight > 100:
        return {
            "success": False,
            "message": f"Modifying {ticker} to {new_weight}% would exceed 100% total weight",
            "portfolio": internal_portfolio,
            "action": "modify_weight",
            "ticker": ticker,
            "new_weight": new_weight
        }
    
    # Update weight while preserving tags
    if isinstance(internal_portfolio[ticker], dict):
        # Has tags - update weight property and preserve all tags
        internal_portfolio[ticker]["weight"] = new_weight
        print(f"🔍 Debug: Updated {ticker} weight to {new_weight}%, preserved tags: {internal_portfolio[ticker]}")
    else:
        # No tags - just update weight
        internal_portfolio[ticker] = new_weight
        print(f"🔍 Debug: Updated {ticker} weight to {new_weight}%")
    
    return {
        "success": True,
        "message": f"Modified {ticker} weight from {current_weight}% to {new_weight}%",
        "portfolio": internal_portfolio,
        "action": "modify_weight",
        "ticker": ticker,
        "new_weight": new_weight
    }

@tool
def get_portfolio_summary(portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get a summary of the current portfolio."""
    # Use provided portfolio or default to empty
    if portfolio is None:
        portfolio = {}
    
    # Convert portfolio to internal format if needed
    if isinstance(portfolio, dict) and "tickers" in portfolio:
        # Frontend format - convert to internal
        internal_portfolio = {}
        for ticker_data in portfolio["tickers"]:
            symbol = ticker_data["symbol"]
            # Preserve the ENTIRE ticker_data structure including tags
            if isinstance(ticker_data, dict) and len(ticker_data) > 2:
                # Has tags - preserve the complete structure
                internal_portfolio[symbol] = ticker_data
            else:
                # No tags - just store weight
                internal_portfolio[symbol] = ticker_data["weight"]
    else:
        # Already internal format
        internal_portfolio = portfolio.copy()
    
    if not internal_portfolio:
        return {
            "success": True,
            "message": "Portfolio is empty",
            "portfolio": internal_portfolio,
            "summary": {
                "ticker_count": 0,
                "total_weight": 0,
                "remaining_weight": 100
            }
        }
    
    # Calculate total weight, handling both simple weights and tagged weights
    total_weight = 0
    for value in internal_portfolio.values():
        if isinstance(value, dict) and "weight" in value:
            weight_value = value["weight"]
        elif isinstance(value, (int, float, str)):
            weight_value = value
        else:
            print(f"⚠️ Warning: Unexpected value type in portfolio: {type(value)}")
            continue
        
        # Convert to float if it's a string
        if isinstance(weight_value, str):
            weight_value = float(weight_value)
        total_weight += weight_value
    
    ticker_count = len(internal_portfolio)
    remaining_weight = 100 - total_weight
    
    summary = {
        "ticker_count": ticker_count,
        "total_weight": total_weight,
        "remaining_weight": remaining_weight,
        "tickers": list(internal_portfolio.keys())
    }
    
    return {
        "success": True,
        "message": f"Portfolio has {ticker_count} tickers with {total_weight}% allocated",
        "portfolio": internal_portfolio,
        "action": "get_summary",
        "summary": summary
    }

@tool
def tag_portfolio_tickers(tag_type: str, portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
    """Add a new tag category to all tickers in the portfolio."""
    # Use provided portfolio or default to empty
    if portfolio is None:
        portfolio = {}
    
    # Normalize tag type (replace spaces with underscores)
    normalized_tag_type = tag_type.replace(" ", "_").lower()
    
    # Convert portfolio to internal format if needed
    if isinstance(portfolio, dict) and "tickers" in portfolio:
        # Frontend format - convert to internal
        internal_portfolio = {}
        for ticker_data in portfolio["tickers"]:
            symbol = ticker_data["symbol"]
            # Preserve the ENTIRE ticker_data structure including tags
            if isinstance(ticker_data, dict) and len(ticker_data) > 2:
                # Has tags - preserve the complete structure
                internal_portfolio[symbol] = ticker_data
            else:
                # No tags - just store weight
                internal_portfolio[symbol] = ticker_data["weight"]
    else:
        # Already internal format
        internal_portfolio = portfolio.copy()
    
    # Initialize tag structure for all tickers
    updated_portfolio = {}
    for symbol, weight in internal_portfolio.items():
        if isinstance(weight, dict):
            # Already has tags - add new tag while preserving existing ones
            existing_ticker = weight.copy()
            existing_ticker[normalized_tag_type] = "AI_Classification_Needed"
            updated_portfolio[symbol] = existing_ticker
            print(f"🔍 Debug: Added {normalized_tag_type} tag to {symbol}, preserved existing tags: {existing_ticker}")
        else:
            # No tags yet - create tag structure
            updated_portfolio[symbol] = {
                "weight": weight,
                normalized_tag_type: "AI_Classification_Needed"
            }
            print(f"🔍 Debug: Created new tag structure for {symbol}: {updated_portfolio[symbol]}")
    
    return {
        "success": True,
        "message": f"Added {tag_type} tag to all tickers",
        "portfolio": updated_portfolio,
        "action": "tag_tickers",
        "tag_type": normalized_tag_type
    }

class PortfolioAgent:
    """Portfolio management agent using LangGraph workflow."""
    
    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initialize the PortfolioAgent with the specified model."""
        self.model_name = model_name
        
        # Initialize tools
        self.tools = [
            add_ticker_to_portfolio,
            remove_ticker_from_portfolio,
            modify_ticker_weight,
            get_portfolio_summary,
            tag_portfolio_tickers
        ]
        
        # Create LLM with tool calling capabilities
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def _classify_tickers_with_ai(self, tag_type: str, ticker_data: List[Dict], internal_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to classify tickers based on the requested tag type."""
        try:
            print(f"🔍 Debug: _classify_tickers_with_ai called with tag_type: {tag_type}")
            print(f"🔍 Debug: ticker_data: {ticker_data}")
            print(f"🔍 Debug: internal_portfolio: {internal_portfolio}")
            
            # Extract ticker symbols for classification
            ticker_symbols = [ticker["symbol"] for ticker in ticker_data]
            
            # Create classification prompt
            classification_prompt = f"""You are a financial expert. Classify the following ticker symbols by their {tag_type}.

Ticker symbols: {', '.join(ticker_symbols)}

For each ticker, provide the most appropriate {tag_type} classification. Use common financial knowledge.

Return ONLY a JSON object with ticker symbols as keys and classifications as values. Example:  
{{
    "SPY": "Equity",
    "BND": "Fixed Income",
    "VTI": "Equity"
}}

No explanations, just the JSON object."""
            
            print(f"🔍 Debug: Sending classification prompt to AI: {classification_prompt}")
            
            # Get AI classification
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            ai_response = response.content
            
            print(f"🔍 Debug: AI response: {ai_response}")
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    classifications = json.loads(json_match.group())
                else:
                    classifications = {}
            except Exception as e:
                print(f"❌ Error parsing classifications: {str(e)}")
                classifications = {}
            
            print(f"🔍 Debug: Parsed classifications: {classifications}")
            
            # Update portfolio with classifications
            new_internal = {}
            for ticker in ticker_data:
                symbol = ticker["symbol"]
                weight = ticker["weight"]
                
                # Get classification for this ticker
                classification = classifications.get(symbol, "Unknown")
                
                # Check if this ticker already exists in the internal portfolio with tags
                if symbol in internal_portfolio and isinstance(internal_portfolio[symbol], dict):
                    # Merge new tag with existing tags
                    existing_ticker = internal_portfolio[symbol].copy()
                    existing_ticker[tag_type] = classification
                    existing_ticker["weight"] = weight  # Update weight if changed
                    new_internal[symbol] = existing_ticker
                    print(f"🔍 Debug: Merged new tag with existing tags for {symbol}: {existing_ticker}")
                else:
                    # Create new ticker data with classification
                    new_ticker_data = {
                        "weight": weight,
                        tag_type: classification
                    }
                    new_internal[symbol] = new_ticker_data
                    print(f"🔍 Debug: Created new ticker data for {symbol}: {new_ticker_data}")
            
            print(f"🔍 Debug: Final new_internal: {new_internal}")
            
            return new_internal
            
        except Exception as e:
            print(f"❌ Error in _classify_tickers_with_ai: {str(e)}")
            return internal_portfolio
    
    def _convert_to_frontend_format(self, internal_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal portfolio format to frontend format."""
        try:
            tickers = []
            total_weight = 0.0
            available_tags = set()
            
            print(f"🔍 Debug: Converting portfolio: {internal_portfolio}")
            
            # Check if we received frontend format instead of internal format
            if isinstance(internal_portfolio, dict) and "tickers" in internal_portfolio:
                print(f"⚠️ Warning: Received frontend format, extracting tickers")
                # Extract the actual portfolio data from frontend format
                for ticker_data in internal_portfolio["tickers"]:
                    if isinstance(ticker_data, dict) and "symbol" in ticker_data and "weight" in ticker_data:
                        tickers.append(ticker_data)
                        weight = float(ticker_data["weight"]) if isinstance(ticker_data["weight"], str) else ticker_data["weight"]
                        total_weight += weight
                        # Check for tags
                        for key, value in ticker_data.items():
                            if key not in ["symbol", "weight"]:
                                available_tags.add(key)
                
                return {
                    "tickers": tickers,
                    "tickerCount": len(tickers),
                    "totalWeight": total_weight,
                    "remainingWeight": 100.0 - total_weight,
                    "isFullyAllocated": abs(total_weight - 100.0) < 0.01,
                    "isOverAllocated": total_weight > 100.0,
                    "tags": list(available_tags)
                }
            
            # Normal internal format processing
            for symbol, data in internal_portfolio.items():
                print(f"🔍 Debug: Processing symbol {symbol} with data {data}")
                
                if isinstance(data, dict):
                    # Has tags
                    if "weight" in data:
                        ticker_info = {
                            "symbol": symbol,
                            "weight": data["weight"]
                        }
                        # Add all tag properties
                        for key, value in data.items():
                            if key not in ["weight"]:
                                ticker_info[key] = value
                                available_tags.add(key)
                        tickers.append(ticker_info)
                        # Ensure weight is converted to float
                        weight = float(data["weight"]) if isinstance(data["weight"], str) else data["weight"]
                        total_weight += weight
                    else:
                        print(f"⚠️ Warning: No weight found for {symbol}, skipping")
                        continue
                elif isinstance(data, (int, float, str)):
                    # Simple weight only
                    tickers.append({
                        "symbol": symbol,
                        "weight": data
                    })
                    # Ensure weight is converted to float
                    weight = float(data) if isinstance(data, str) else data
                    total_weight += weight
                else:
                    print(f"⚠️ Warning: Unexpected data type for {symbol}: {type(data)}, skipping")
                    continue
            
            print(f"🔍 Debug: Converted tickers: {tickers}")
            print(f"🔍 Debug: Total weight: {total_weight}")
            print(f"🔍 Debug: Available tags: {available_tags}")
            
            return {
                "tickers": tickers,
                "tickerCount": len(tickers),
                "totalWeight": total_weight,
                "remainingWeight": 100.0 - total_weight,
                "isFullyAllocated": abs(total_weight - 100.0) < 0.01,
                "isOverAllocated": total_weight > 100.0,
                "tags": list(available_tags)
            }
            
        except Exception as e:
            print(f"❌ Error converting to frontend format: {str(e)}")
            import traceback
            print(f"🔍 Debug: Full traceback: {traceback.format_exc()}")
            return {
                "tickers": [],
                "tickerCount": 0,
                "totalWeight": 0.0,
                "remainingWeight": 100.0,
                "isFullyAllocated": False,
                "isOverAllocated": False,
                "tags": []
            }
    
    def _build_graph(self):
        """Build the LangGraph workflow using StateGraph and ToolNode."""
        # Create the graph builder
        builder = StateGraph(PortfolioState)
        
        # Define the agent node that decides what to do
        def agent_node(state: PortfolioState):
            """Node that processes user input and decides on actions."""
            messages = state["messages"]
            portfolio = state.get("portfolio", {})
            current_tool_results = state.get("current_tool_results", [])
            
            print(f"🔍 Debug: Agent node received portfolio state: {portfolio}")
            print(f"🔍 Debug: Current tool results: {current_tool_results}")
            
            # If we have tool results, the agent should respond to them
            if current_tool_results:
                # Create response to tool results
                response_prompt = f"""You are a portfolio management assistant. The following tools have been executed:

{chr(10).join([f"- {result.get('action', 'unknown')}: {result.get('message', 'No message')}" for result in current_tool_results])}

Current portfolio state: {portfolio}

Provide a clear, concise response about what was accomplished. If the portfolio was modified, show the current state using format "TICKER: WEIGHT%" for each ticker.

Keep your response under 100 words and be helpful."""
                
                response = self.llm.invoke([HumanMessage(content=response_prompt)])
                
                # Clear tool results and mark as needing final response
                return {
                    "messages": [response],
                    "current_tool_results": [],
                    "needs_final_response": True
                }
            
            # Create system prompt for new user requests
            system_prompt = f"""You are a portfolio management assistant. You can:
1. Add new tickers with weights
2. Remove existing tickers
3. Modify ticker weights
4. Show portfolio summary
5. Tag portfolio tickers with specific categories (e.g., asset class, region, sector)

When a user asks to:
- Add a ticker: Use add_ticker_to_portfolio with ticker symbol and weight
- Remove a ticker: Use remove_ticker_from_portfolio with ticker symbol
- Modify a ticker: Use modify_ticker_weight with ticker symbol and new weight
- Show portfolio: Use get_portfolio_summary
- Tag portfolio: Use tag_portfolio_tickers with tag_type (e.g., "asset_class", "region", "sector")

CRITICAL: When modifying ticker weights or rebalancing, you MUST use the modify_ticker_weight tool for each ticker change to preserve existing tags. Do NOT create new portfolio structures manually.

REBALANCING RULE: If a user asks to "rebalance" or "modify and rebalance", you MUST:
1. Calculate the new weights for each ticker
2. Call modify_ticker_weight for EACH ticker individually
3. NEVER create portfolio structures manually in your response

Always ensure the total portfolio weight doesn't exceed 100%. If a request would exceed this limit, explain why it can't be done.

Current portfolio: {portfolio}

IMPORTANT: Each tool call will automatically receive the current portfolio state with all existing tags and modifications. You only need to provide the specific parameters (ticker, weight, tag_type). The portfolio state is maintained between tool calls.

Respond naturally and call the appropriate tool when needed."""
            
            # Get AI response with tool calls
            response = self.llm_with_tools.invoke([
                HumanMessage(content=system_prompt),
                *messages
            ])
            
            return {"messages": [response]}
        
        # Define a custom tool execution node that properly handles state
        def tool_execution_node(state: PortfolioState):
            """Execute tools with proper state management."""
            messages = state["messages"]
            last_message = messages[-1]
            portfolio = state.get("portfolio", {})
            
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return {"messages": [last_message]}
            
            tool_results = []
            updated_portfolio = portfolio.copy()
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"🔍 Debug: Executing tool: {tool_name}")
                print(f"🔍 Debug: Tool args: {tool_args}")
                print(f"🔍 Debug: Current portfolio: {updated_portfolio}")
                
                # Find and execute the tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        # Always pass the current updated portfolio to tools
                        tool_args["portfolio"] = updated_portfolio
                        print(f"🔍 Debug: Passing portfolio to {tool_name}: {updated_portfolio}")
                        
                        # Execute the tool with portfolio
                        result = tool.invoke(tool_args)
                        print(f"🔍 Debug: Tool result: {result}")
                        
                        if result.get("success", False):
                            # Update portfolio with tool result
                            updated_portfolio = result["portfolio"]
                            print(f"🔍 Debug: Updated portfolio after {tool_name}: {updated_portfolio}")
                            
                            # Special handling for tag_portfolio_tickers
                            if tool_name == "tag_portfolio_tickers":
                                # Extract tag type from the tool args
                                tag_type = tool_args.get("tag_type", "unknown")
                                print(f"🔍 Debug: Starting AI classification for tag: {tag_type}")
                                
                                # Convert the updated portfolio to ticker data format for classification
                                ticker_data = []
                                for symbol, data in updated_portfolio.items():
                                    if isinstance(data, dict):
                                        ticker_data.append({
                                            "symbol": symbol,
                                            "weight": data["weight"]
                                        })
                                    else:
                                        ticker_data.append({
                                            "symbol": symbol,
                                            "weight": data
                                        })
                                
                                updated_portfolio = self._classify_tickers_with_ai(
                                    tag_type,
                                    ticker_data,
                                    updated_portfolio
                                )
                                print(f"🔍 Debug: AI classification result: {updated_portfolio}")
                        
                        tool_results.append(result)
                        break
            
            # Create tool messages
            tool_messages = []
            for i, tool_call in enumerate(last_message.tool_calls):
                if i < len(tool_results):
                    result = tool_results[i]
                    tool_message = ToolMessage(
                        content=str(result.get("message", "Tool executed")),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                    tool_messages.append(tool_message)
            
            # Update state with new portfolio and tool results
            print(f"🔍 Debug: Tool execution node returning portfolio: {updated_portfolio}")
            return {
                "messages": tool_messages,
                "portfolio": updated_portfolio,
                "current_tool_results": tool_results
            }
        
        # Define the condition for routing
        def should_continue(state: PortfolioState):
            """Determine whether to continue with tools or end."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the last message has tool calls, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # If we need a final response, end
            if state.get("needs_final_response", False):
                return END
            
            return END
        
        # Add nodes to the graph
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_execution_node)
        
        # Add edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, ["tools", END])
        builder.add_edge("tools", "agent")
        
        # Compile the graph
        return builder.compile()
    
    def chat(self, user_message: str, portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main chat method that uses the LangGraph workflow."""
        try:
            # Convert frontend portfolio to internal format
            if portfolio and "tickers" in portfolio:
                internal_portfolio = {}
                for ticker_data in portfolio["tickers"]:
                    symbol = ticker_data["symbol"]
                    # Preserve the ENTIRE ticker_data structure including tags
                    if isinstance(ticker_data, dict) and len(ticker_data) > 2:
                        # Has tags - preserve the complete structure
                        internal_portfolio[symbol] = ticker_data
                    else:
                        # No tags - just store weight
                        internal_portfolio[symbol] = ticker_data["weight"]
            else:
                internal_portfolio = portfolio or {}
            
            print(f"🔍 Debug: Initial portfolio for chat: {internal_portfolio}")
            
            # Prepare initial state for LangGraph with all required fields
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "portfolio": internal_portfolio,
                "current_tool_results": [],
                "needs_final_response": False
            }
            
            print(f"🔍 Debug: Initial state: {initial_state}")
            
            # Execute the LangGraph workflow
            result = self.graph.invoke(initial_state)
            
            print(f"🔍 Debug: Final result state: {result}")
            
            # Extract the final response and portfolio
            final_messages = result.get("messages", [])
            final_portfolio = result.get("portfolio", internal_portfolio)
            
            # Get the last meaningful message
            final_message = None
            for msg in reversed(final_messages):
                if hasattr(msg, 'content') and msg.content:
                    final_message = msg
                    break
            
            if not final_message:
                final_message = final_messages[-1] if final_messages else HumanMessage(content="No response generated")
            
            # Convert final portfolio to frontend format
            frontend_portfolio = self._convert_to_frontend_format(final_portfolio)
            
            return {
                "response": final_message.content,
                "portfolio_state": frontend_portfolio,
                "changes": ["Portfolio was updated through LangGraph workflow"]
            }
                
        except Exception as e:
            # Fallback response if processing fails
            error_msg = f"I encountered an error processing your request: {str(e)}"
            print(f"❌ Error in chat method: {str(e)}")
            import traceback
            print(f"🔍 Debug: Full traceback: {traceback.format_exc()}")
            
            return {
                "response": error_msg,
                "portfolio_state": self._convert_to_frontend_format(portfolio or {}),
                "changes": []
            }

# Create a global instance for easy access
portfolio_agent = PortfolioAgent()

# Export the main function for external use
def chat_with_portfolio_agent(user_message: str, portfolio: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to chat with the portfolio agent.
    
    Args:
        user_message: The user's input message
        portfolio: Current portfolio state (optional)
        
    Returns:
        Dict containing response, portfolio_state, and changes
    """
    return portfolio_agent.chat(user_message, portfolio)

if __name__ == "__main__":
    # Test the agent
    test_portfolio = {"AAPL": 30, "GOOGL": 25, "MSFT": 20}
    
    print("Portfolio Agent Test")
    print("=" * 50)
    
    test_messages = [
        "What's in my portfolio?",
        "Add TSLA with 15% weight",
        "Remove MSFT from my portfolio",
        "How is my portfolio looking?"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        result = portfolio_agent.chat(message, test_portfolio)
        print(f"Response: {result['response']}")
        print(f"Portfolio: {result['portfolio_state']}")
        print(f"Changes: {result['changes']}")
        print("-" * 30)
