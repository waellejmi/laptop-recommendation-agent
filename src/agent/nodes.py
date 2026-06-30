import json
import logging
import os
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.types import Command

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import AnyUrl

from agent.prompts import (
    CLASSIFICATION_PROMPT,
    EXPLANATION_PROMPT,
    EXPLANATION_PROMPT_GAME,
    MAPPING_PROMPT,
)
from agent.state import GameSpecification, SalesAgentState, UserRequestClassification
from filter import filter_laptops
from scoring import compute_scores, get_weights
from tools.sys_req_lookup_tool import (
    GameNotFound,
    local_lookup,
)

load_dotenv()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SalesAgent:
    def __init__(self, provider: str = "groq"):
        if provider == "groq":
            GROQ_API_KEY = os.getenv("GROQ_KEY")
            self.llm = ChatGroq(
                model="qwen/qwen3-32b", api_key=GROQ_API_KEY, temperature=0.5
            )
        elif provider == "huggingface":
            HF_TOKEN = os.getenv("HF_TOKEN")
            self.llm = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    task="text-generation",
                    max_new_tokens=1024,
                    do_sample=False,
                    repetition_penalty=1.03,
                    temperature=0.5,
                    huggingfacehub_api_token=HF_TOKEN,
                ),
                verbose=True,
            )
        self._game_db_path: Optional[str] = None
        self._laptop_db_path: Optional[str] = None
        self.server_params = StdioServerParameters(
            command="python", args=["mcp_server.py"]
        )

    # MCP SETup
    async def _get_resource_path_from_session(self, session, uri: str) -> str:
        result = await session.read_resource(AnyUrl(uri))
        raw_text = result.contents[0].text
        logger.info(f"Resource info for {uri}: {raw_text}")
        resource_info = json.loads(raw_text)
        return resource_info["path"]

    async def get_mcp_paths(self):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                resources = await session.list_resources()
                logger.info("MCP CONNECTED")
                logger.info("Available MCP Resources:")
                for resource in resources.resources:
                    logger.info(f"  - {resource.name}: {resource.uri}")
                self._game_db_path = await self._get_resource_path_from_session(
                    session, "file:///data/games-system-requirements/game_db.csv"
                )
                self._laptop_db_path = await self._get_resource_path_from_session(
                    session, "file:///data/laptops_enhanced.csv"
                )

    async def online_lookup_mcp(self, game_name: str) -> dict:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()

                logger.info("Available MCP Tools:")
                for tool in tools.tools:
                    logger.info(f"  - {tool.name}: {tool.description}")
                result = await session.call_tool(
                    "online_lookup", arguments={"game_name": game_name}
                )
                return json.loads(result.content[0].text)

    # Agent logic and nodes
    def read_user_prompt(self, state: SalesAgentState) -> dict:
        # maybe i add read from a text file or live from a cli
        return {
            "messages": [
                HumanMessage(content=f"Processing User Input: {state['user_input']}")
            ]
        }

    def classify_intent(
        self,
        state: SalesAgentState,
    ) -> Command[
        Literal[
            "get_filtered_laptops",
            "get_recommended_laptops",
            "get_game_requirements",
            "inform_user_gibberish",
        ]
    ]:
        structured_llm = self.llm.with_structured_output(UserRequestClassification)

        classification_prompt = CLASSIFICATION_PROMPT.format(
            user_input=state["user_input"]
        )

        classification = structured_llm.invoke(classification_prompt)

        if classification.gibberish:
            goto = "inform_user_gibberish"
        elif classification.specific_game:
            goto = "get_game_requirements"

        elif classification.filters:
            goto = "get_filtered_laptops"
        else:
            goto = "get_recommended_laptops"

        return Command(
            update={"classification": classification.model_dump()}, goto=goto
        )

    async def get_game_requirements(
        self,
        state: SalesAgentState,
    ) -> Command[Literal["get_filtered_laptops", "inform_user_no_laptops"]]:
        classification = state.get("classification", {})
        game_name = classification.get("specific_game", "")
        try:
            try:
                # we first look online, because it faster than querying this 80K entry we have
                full_res = await self.online_lookup_mcp(game_name)

                if not full_res.get("success"):
                    raise GameNotFound("Online lookup failed")

                recc_game_requirements = full_res["data"]
            except GameNotFound:
                # if it fails we check the local db plus local db have only minimum requirments
                recc_game_requirements = local_lookup(
                    game_name, path=self._game_db_path
                )
            structured_llm = self.llm.with_structured_output(GameSpecification)

            mapping_prompt = MAPPING_PROMPT.format(
                game_name=recc_game_requirements["game_name"],
                gpu=recc_game_requirements["gpu"],
                cpu=recc_game_requirements["cpu"],
            )

            game_specific_filters = structured_llm.invoke(mapping_prompt)
            return Command(
                update={
                    "game_system_requirements": recc_game_requirements,
                    "game_specific_filters": game_specific_filters.model_dump(),
                },
                goto="get_filtered_laptops",
            )
        except GameNotFound as e:
            return Command(
                update={
                    "final_response": f"{str(e)}",
                },
                goto="inform_user_no_laptops",
            )

    def get_filtered_laptops(
        self,
        state: SalesAgentState,
    ) -> Command[Literal["get_recommended_laptops", "inform_user_no_laptops"]]:
        classification = state.get("classification", {})
        game_requirements = state.get("game_specific_filters", {})
        filters = classification.get("filters") or {}

        if isinstance(game_requirements, dict) and game_requirements.get("ram"):
            filters["ram"] = game_requirements["ram"]
        if isinstance(game_requirements, dict) and game_requirements.get("gpu"):
            filters["gpu"] = game_requirements["gpu"]
            filters["sort_by_gpu_tier"] = True
        try:
            # filtered_laptops_df = filter_laptops(**filters)
            filtered_laptops_df = filter_laptops(
                dataset_path=self._laptop_db_path, **filters
            )
            if filtered_laptops_df.empty:
                return Command(
                    update={"filtered_laptops": []}, goto="inform_user_no_laptops"
                )
            filtered_laptops = filtered_laptops_df.to_dict(orient="records")
            return Command(
                update={"filtered_laptops": filtered_laptops},
                goto="get_recommended_laptops",
            )
        except Exception as e:
            return Command(
                update={
                    "filtered_laptops": [],
                    "final_response": f"Filtering Error: {str(e)}",
                },
                goto="inform_user_no_laptops",
            )

    def get_recommended_laptops(
        self,
        state: SalesAgentState,
    ) -> Command[Literal["explain_reccomandations"]]:
        classification = state.get("classification", {})
        usage_profile = classification.get("usage_profile", "basic")
        user_emphasis = classification.get("user_emphasis", [])
        filtered_laptops = state.get("filtered_laptops", list[dict])

        profile_weights = get_weights(usage_profile, user_emphasis)

        if filtered_laptops is None or len(filtered_laptops) == 0:
            df = pd.read_csv("./data/laptops_enhanced.csv")
        else:
            df = pd.DataFrame(filtered_laptops)

        top3_ranked = compute_scores(df, profile_weights).head(3)
        recommended_laptops = top3_ranked.to_dict(orient="records")

        return Command(
            update={"recommended_laptops": recommended_laptops},
            goto="explain_reccomandations",
        )

    def explain_reccomandations(
        self,
        state: SalesAgentState,
    ) -> Command[Literal["send_reply"]]:
        reccomended_laptops = state.get("recommended_laptops", [])
        user_context = state.get("user_input", "")
        game_requirements = state.get("game_system_requirements", {})

        if isinstance(game_requirements, dict) and game_requirements.get("game_name"):
            explanation_prompt = EXPLANATION_PROMPT_GAME.format(
                game_name=game_requirements["game_name"],
                reccomended_laptops=reccomended_laptops,
                user_context=user_context,
                gpu=game_requirements["gpu"],
                cpu=game_requirements["cpu"],
                ram=game_requirements["ram"],
            )
        else:
            explanation_prompt = EXPLANATION_PROMPT.format(
                reccomended_laptops=reccomended_laptops,
                user_context=user_context,
            )

        explanation = self.llm.invoke(explanation_prompt)

        return Command(
            update={"final_response": explanation.content},
            goto="send_reply",
        )

    @staticmethod
    def inform_user_no_laptops(_state: SalesAgentState) -> dict:
        return {
            "final_response": (
                "Unfortunately, no laptops match your specified criteria in our database."
                "Please consider adjusting your requirements."
            )
        }

    @staticmethod
    def inform_user_gibberish(_state: SalesAgentState) -> dict:
        return {
            "final_response": (
                "Unfortunately, we did not understand your request. Please stay on topic of laptops."
                "I am here to help you find a Laptop specific to your needs."
            )
        }

    @staticmethod
    def send_reply(state: SalesAgentState) -> dict:
        print("\n" + "=" * 80)
        print("AGENT RESPONSE:")
        print("=" * 80)
        print(state["final_response"])
        print("=" * 80 + "\n")
        return {}
