import asyncio
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from app.states.agent_state import AgentState

# Assuming we have an LLM configured in our project
from app.llms.openai_llm import llm

async def retrieve_content_from_mcp(query: str, server_url: str = "http://127.0.0.1:8003/sse") -> str:
    """
    Connects to an MCP server via SSE, loads its available tools,
    and uses them to retrieve context/content.
    """
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection to the MCP server
            await session.initialize()
            
            # Fetch available tools from the MCP server
            mcp_tools = await load_mcp_tools(session)
            
            # Create a ReAct agent powered by our LLM and the MCP tools
            system_prompt = (
                "You are a documentation retriever. "
                "Use the search tool ONCE to find relevant documentation for the given code. "
                "From the results, pick only the TOP 2 most relevant excerpts (highest score). "
                "Return a concise summary (3-5 sentences max) of only the critical rules, "
                "constraints, or API behaviours relevant to the code. "
                "Do NOT dump raw text. Do NOT repeat yourself."
            )
            agent = create_react_agent(llm, tools=mcp_tools, prompt=system_prompt)
            
            print(f"\n[*] Connected to MCP Server. Loaded {len(mcp_tools)} tools.")
            
            # Run the agent and stream events to capture intermediate tool calls/results
            final_content = ""
            async for event in agent.astream({"messages": [HumanMessage(content=query)]}):
                # Check if it's a tool event containing responses from the MCP server
                if "tools" in event:
                    for msg in event["tools"]["messages"]:
                        print("\n[MCP Server Data / Tool Output]")
                        print(f"Tool Name: {msg.name}")
                        print(f"Content:\n{msg.content}\n")
                        print("-" * 50)
                
                # Check if it's the final agent generation
                if "agent" in event:
                    for msg in event["agent"]["messages"]:
                        if msg.content:
                            final_content = msg.content

            return final_content

async def context_retriever_node(state: AgentState) -> dict:
    """
    Use this node to connect to the external MCP server and retrieve content or context.
    Updates the 'context' field in the state.
    """
    code = state.get("code", "")
    
    # Instruct the agent to find context related to the code
    query = f"Search the documents and return relevant documentation for the following code:\n{code}"
        
    result = await retrieve_content_from_mcp(query)
    return {"context": result}


