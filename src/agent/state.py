from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class LaptopSpecification(BaseModel):
    model_name: Optional[str] = None
    brand: Optional[str] = None
    cpu: Optional[str] = None
    cpu_cores: Optional[int] = None
    cpu_threads: Optional[int] = None
    ram: Optional[int] = None
    ssd_gb: Optional[int] = None
    hdd_gb: Optional[int] = None
    os: Optional[str] = None
    gpu: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    screen_size_in: Optional[float] = None
    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    resolution_type: Optional[str] = None
    sort_by_gpu_tier: Optional[bool] = False
    price_euro: Optional[float] = None


class GameSpecification(BaseModel):
    ram: Optional[int] = None
    gpu: Optional[str] = None


class UserRequestClassification(BaseModel):
    usage_profile: Literal["gaming", "student", "basic", "workstation"]
    user_emphasis: Optional[
        list[Literal["cpu_tier", "gpu_tier", "ram", "ssd_present", "price"]]
    ] = None
    filters: Optional[LaptopSpecification] = None
    specific_game: Optional[str] = None
    gibberish: Optional[bool] = False


class SalesAgentState(TypedDict):
    user_input: str

    classification: dict[str, Any] | None

    game_specific_filters: dict[str, Any] | None
    game_system_requirements: dict[str, Any] | None

    filtered_laptops: list[dict[str, Any]] | None

    recommended_laptops: list[dict[str, Any]] | None

    final_response: str | None

    messages: Annotated[list[AnyMessage], add_messages]
