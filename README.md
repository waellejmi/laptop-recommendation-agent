# Laptop Recommendation Agent

As the computer guy of the family, everyone who is buying a laptop ends up calling me and asking which brand or model to get. That is fair. Laptops are genuinely confusing for non-technical people, and specs alone do not explain whether a device actually fits someone’s needs.

This project was my way of turning that recurring situation into a recommendation system, while also introducing myself to recommender systems where there is no prior customer knowledge.

## Core Approach

The user simply explains what they want in natural language.  
The LLM maps that input to:
- a usage profile selected from a fixed set
- optional emphasis to adjust scoring priorities
- hard constraints such as budget, brand, or screen size

Each laptop is then scored using profile-specific weights, rather than generic ranking logic.


## Profiles and Filtering

Supported profiles include gaming, student, basic/office, and workstation.  
Hard filters are applied only when explicitly mentioned, ensuring flexibility without ignoring user constraints.

## Game-Aware Matching

I added a fun but practical feature for gaming use cases:
- If the user mentions a game, the agent searches the web and scrapes system requirements
- If scraping fails, it falls back to a local dataset
- As a final fallback, it uses a large Kaggle game requirements database with ~80k entries

The extracted requirements are compared directly against the laptop database to ensure smooth gameplay.

## Orchestration and MCP

The entire flow is orchestrated using LangGraph.

Later, I decided to integrate MCP, mostly out of curiosity. This definitely overcomplicated a simple app, but it was worth it as a learning exercise. MCP allowed:
- databases to be exposed as resources
- tools to be decoupled from agent logic
- easier future expansion without code changes

Two resources were added: the laptop database and the local game requirements database.  
The online search tool was moved to an MCP tool.

The MCP server runs over stdio to reduce networking overhead since everything is local.

This required refactoring a significant portion of the codebase, but seeing the full system working end to end made it worthwhile.

For a fully working version without MCP, see the older commit labeled **Small fixup.”**

## Examples and Results

You can find example prompts and outputs [here](https://github.com/waellejmi/laptop-sales-agent/tree/main/results)  

## Trying It Out

If you want to run it locally:
- Create a `.env` file and set a `GROQ_KEY` API key  
- HuggingFace key is optional
- Run `uv sync`
- Start the CLI

## Datasets

Laptop dataset (enhanced):  
https://www.kaggle.com/datasets/waellejmi/laptops-specificaitons-enhanced/data

Original dataset source:  
https://www.kaggle.com/datasets/sushmita36/laptops-dataset

Game requirements dataset (fallback):  
https://www.kaggle.com/datasets/hatoonaloqaily/steam-system-requirements-dataset
