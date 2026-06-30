# cli.py
import asyncio
import sys

from agent import build_graph
from tests import test_agent_with_streaming


async def run_cli(debug=False):
    walfred = await build_graph()

    print("=" * 80)
    print("LAPTOP SALES AGENT - Walfred")
    print("=" * 80)
    print("Commands:")
    print("  'quit' or 'q' - Exit")
    print("  'debug' - Toggle debug mode")
    print("  'clear' - Clear screen")
    print("=" * 80 + "\n")

    session_count = 0

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if user_input.lower() == "debug":
            debug = not debug
            print(f"\nDebug mode: {'ON' if debug else 'OFF'}")
            continue

        if user_input.lower() == "clear":
            print("\n" * 50)
            continue

        if not user_input:
            continue

        session_count += 1
        config = {"configurable": {"thread_id": f"session_{session_count}"}}

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

        try:
            if debug:
                await test_agent_with_streaming(user_input)
            else:
                await walfred.ainvoke(initial_state, config)

        except Exception as e:
            print(f"\nError: {str(e)}")
            if debug:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    debug_mode = "--debug" in sys.argv
    asyncio.run(run_cli(debug=debug_mode))
