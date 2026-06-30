import asyncio
from time import sleep

from src.agent.graph import build_graph


async def test_agent(user_input: str):
    walfred = await build_graph(provider="groq")
    config = {"configurable": {"thread_id": "test_session_1"}}
    initial_state = {
        "user_input": user_input,
        "classification": None,
        "filtered_laptops": None,
        "game_specific_filters": None,
        "game_system_requirements": None,
        "recommended_laptops": None,
        "final_response": None,
        "messages": [],
    }

    result = await walfred.ainvoke(initial_state, config)
    return result


async def test_agent_with_streaming(user_input: str):
    walfred = await build_graph(provider="groq")
    config = {"configurable": {"thread_id": "test_session_1"}}

    initial_state = {
        "user_input": user_input,
        "classification": None,
        "game_system_requirements": None,
        "game_specific_filters": None,
        "filtered_laptops": None,
        "recommended_laptops": None,
        "final_response": None,
        "messages": [],
    }

    print(f"\n{'=' * 80}")
    print(f"USER INPUT: {user_input}")
    print(f"{'=' * 80}\n")

    async for event in walfred.astream(initial_state, config):
        node_name = list(event.keys())[0]
        node_state = event[node_name]

        if not node_state:
            print(f"\n--- NODE: {node_name} (no state update) ---")
            continue

        print(f"\n--- NODE: {node_name} ---")

        classification = node_state.get("classification")
        if classification:
            print(f"Classification: {classification}")

        game_system_requirements = node_state.get("game_system_requirements")
        if game_system_requirements is not None:
            if isinstance(game_system_requirements, dict):
                print(f"Game System Requirements: {game_system_requirements} ")
            game_specific_filters = node_state.get("game_specific_filters")
            if isinstance(game_specific_filters, dict):
                print(f"Game Specific Filters: {game_specific_filters} ")

        filtered = node_state.get("filtered_laptops")
        if filtered is not None:
            if isinstance(filtered, list):
                print(f"Filtered Laptops: {len(filtered)} found")
            else:
                print(f"Filtered Laptops: {filtered}")

        recommended = node_state.get("recommended_laptops")
        if recommended:
            print(f"Recommendations: {len(recommended)}")
            for i, laptop in enumerate(recommended, 1):
                print(
                    f"  {i}. {laptop.get('model_name')} - €{laptop.get('price_euro')}"
                )

        final_response = node_state.get("final_response")
        if final_response:
            print(f"\nFinal Response: {final_response[:200]}...")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    prompts = [
        "I need the best gaming laptop under 3000 euros. I need to play the latest games in 4K smoothly",
        "I'm searching exactly for this laptop Lenovo IdeaPad Gaming 3 82K201UEIN. Do you have it in your store?",
        "I am going to uni this year and I need a lapoptop for taking notes and doing assignments. Money is most important factor for me. I need best price to performance ratio. My budget is 800 euros, i can go a bit higher like 100 extra euro max",
        "je veux un ordinateur pour jouer god of war j ai un budget de 1000 euro et je veux un pc asus parceque il sont tres fiable",
    ]
    # print("# TEST 1: Gaming Laptop")
    # asyncio.run(test_agent(prompts[0]))
    # sleep(2)  # Groq limit rate so i have to slow down. call me sonic
    print("\n # TEST 2: Exact Model")
    asyncio.run(test_agent(prompts[1]))
    sleep(2)  # Groq limit rate so i have to slow down. call me sonic
    print("\n # TEST 3: Very Specific Student Needs")
    asyncio.run(test_agent(prompts[2]))
    sleep(2)  # Groq limit rate so i have to slow down. call me sonic
    print("\n # TEST 4: MultiLangual Test")
    asyncio.run(test_agent(prompts[3]))
