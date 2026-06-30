from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import SalesAgent
from agent.state import SalesAgentState


async def build_graph(provider: str = "groq"):
    agent = SalesAgent(provider)
    await agent.get_mcp_paths()

    workflow = StateGraph(SalesAgentState)
    workflow.add_node("read_user_prompt", agent.read_user_prompt)
    workflow.add_node("classify_intent", agent.classify_intent)
    workflow.add_node("get_game_requirements", agent.get_game_requirements)
    workflow.add_node("get_filtered_laptops", agent.get_filtered_laptops)
    workflow.add_node("get_recommended_laptops", agent.get_recommended_laptops)
    workflow.add_node("explain_reccomandations", agent.explain_reccomandations)
    workflow.add_node("inform_user_no_laptops", agent.inform_user_no_laptops)
    workflow.add_node("inform_user_gibberish", agent.inform_user_gibberish)
    workflow.add_node("send_reply", agent.send_reply)

    workflow.add_edge(START, "read_user_prompt")
    workflow.add_edge("read_user_prompt", "classify_intent")
    workflow.add_edge("inform_user_no_laptops", "send_reply")
    workflow.add_edge("inform_user_gibberish", "send_reply")
    workflow.add_edge("send_reply", END)

    memory = MemorySaver()

    graph = workflow.compile(checkpointer=memory)
    return graph
