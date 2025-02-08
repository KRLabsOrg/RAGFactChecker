from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ExperimentSetupConfig:
    system_retry: int
    log_prompts: bool


@dataclass
class AnswerGeneratorConfig:
    model_name: str
    num_shot: int


@dataclass
class TripletGeneratorModelParams:
    openie_affinity_probability_cap: float


@dataclass
class TripletGeneratorConfig:
    model_name: str
    model_params: TripletGeneratorModelParams
    num_shot: int


@dataclass
class LLMConfig:
    generator_model: str
    request_max_try: int
    temperature: float
    api_key: Optional[str]


@dataclass
class ModelConfig:
    answer_generator: AnswerGeneratorConfig
    triplet_generator: TripletGeneratorConfig
    llm: LLMConfig


@dataclass
class PathDataConfig:
    base: str
    demo: str


@dataclass
class PathConfig:
    data: PathDataConfig
    prompts: str


@dataclass
class Config:
    experiment_setup: ExperimentSetupConfig
    model: ModelConfig
    path: PathConfig
    logger_level: Optional[str] = None


@dataclass
class TripletGeneratorOutput:
    triplets: List[List[str]]


@dataclass
class FactCheckerOutput:
    fact_check_prediction_binary: Dict[str, bool]


@dataclass
class HallucinationDataGeneratorOutput:
    generated_hlcntn_answer: str
    generated_non_hlcntn_answer: str
    hlcntn_part: str


@dataclass
class DirectTextMatchOutput:
    input_triplets: List[List[str]]
    reference_triplets: List[List[str]]
    fact_check_prediction_binary: Dict[str, bool]
