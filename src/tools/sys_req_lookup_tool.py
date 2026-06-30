import difflib
import logging
import re
from typing import Dict

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GameNotFound(Exception):
    pass


def game_normalize(text) -> str:
    return re.sub(r"\W+", " ", text).lower().strip()


def title_normalize(text) -> str:
    # s = text.lower().strip()
    parts = re.split(r"system requirements", text, flags=re.IGNORECASE)
    return parts[0].strip() if len(parts) > 1 else text


def best_match(game_name, results):
    game_name_norm = game_normalize(game_name)
    best_score = 0
    best_result = None
    best_index = -1
    for idx, result in enumerate(results):
        title_norm = title_normalize(result["title"]).lower().strip()
        score = difflib.SequenceMatcher(None, game_name_norm, title_norm).ratio()
        if score > best_score:
            best_score = score
            best_result = result
            best_index = idx
    if best_score < 0.4:
        raise GameNotFound(f"No good online match found for '{game_name}'")
    return best_result, best_score, best_index


def search_web(game_name: str) -> Dict[str, str]:
    """
    Search the web for a game system requirments using DuckDuckGo and return the systemrequirementslab  and returns top 3 links
    """
    query = f"{game_name} system requirements site:systemrequirementslab.com"
    try:
        search = DuckDuckGoSearchResults(output_format="list", num_results=3)
        results = search.invoke(query)
        return results
    except Exception as e:
        return {"error": f"Error while searching the web: {str(e)}"}


def scrape_from_srl(url: str) -> str:
    """
    Given a URL, fetch the page and extract the system requirements information using a CSS selector.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        selector = ".container > div:nth-child(5) > div:nth-child(4) > div:nth-child(1)"
        req_section = soup.select_one(selector)
        if req_section:
            return req_section.get_text(separator="\n").strip()
        else:
            return "System requirements section not found."
    except Exception as e:
        return f"Error fetching system requirements: {str(e)}"


def online_lookup(game_name: str):
    try:
        results = search_web(game_name)
        result, score, id = best_match(game_name, results)
        logger.info("Found an online match:")
        logger.info(f"Best result: {result}, Score: {score}, Index: {id}")
        output = scrape_from_srl(results[id]["link"])
        cpu = re.search(r"CPU\s*: (.+)", output)
        ram = re.search(r"RAM\s*: (.+)", output)
        gpu = re.search(r"VIDEO CARD\s*: (.+)", output)

        cpu = cpu.group(1) if cpu else "Not found"
        ram = ram.group(1) if ram else "Not found"
        gpu = gpu.group(1) if gpu else "Not found"
        game_system_req = {
            "game_name": title_normalize(result["title"]),
            "cpu": cpu,
            "gpu": gpu,
            "ram": ram,
        }
        return game_system_req
    except GameNotFound as e:
        raise e


def local_lookup(
    game_name: str,
    dataset_path: str = "./data/games-system-requirements/game_db.csv",
) -> Dict[str, str]:
    import pandas as pd

    try:
        logger.info(f"Loading local dataset from {dataset_path}")
        db = pd.read_csv(dataset_path)
        game_name_norm = game_normalize(game_name)
        best_score = 0
        best_result = None
        best_index = -1
        for idx, row in db.iterrows():
            title = str(row.get("name", ""))
            title_norm = title_normalize(title).lower().strip()
            score = difflib.SequenceMatcher(None, game_name_norm, title_norm).ratio()
            if score > best_score:
                best_score = score
                best_result = row.to_dict()
                best_index = idx
        if best_score < 0.4:
            raise GameNotFound(f"No good local match found for '{game_name}'")
        logger.info("Found a local match:")
        logger.info(
            f"Best result: {best_result}, Score: {best_score}, Index: {best_index}"
        )
        return {
            "game_name": title_normalize(best_result.get("name", "")),
            "cpu": best_result.get("cpu", "").strip(),
            "gpu": best_result.get("gpu", "").strip(),
            "ram": best_result.get("ram", "").strip(),
        }

    except GameNotFound as e:
        raise e


def get_system_requirements(game_name: str, online: bool = True) -> Dict[str, str]:
    if online:
        return online_lookup(game_name)
    return local_lookup(game_name)


if __name__ == "__main__":
    game_name = "Hogwarts Lgac"
    print(local_lookup(game_name))
    print(online_lookup(game_name))
    # print(get_system_requirements(game_name))
