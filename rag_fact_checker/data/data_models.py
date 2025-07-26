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
class FactCheckerConfig:
    inquiry_mode: bool = False
    split_reference_triplets: bool = False


@dataclass
class LLMConfig:
    generator_model: str
    request_max_try: int
    temperature: float
    api_key: str | None


@dataclass
class SimpleBatchConfig:
    """Configuration for simple batch processing."""

    max_workers: int = 5  # Number of concurrent threads
    max_retries: int = 3  # Maximum retries for failed items
    retry_delay: float = 1.0  # Delay between retries in seconds
    timeout: float | None = None  # Timeout per individual call


@dataclass
class ModelConfig:
    answer_generator: AnswerGeneratorConfig
    triplet_generator: TripletGeneratorConfig
    fact_checker: FactCheckerConfig
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
    logger_level: str | None = None
    simple_batch_config: SimpleBatchConfig | None = None


@dataclass
class TripletGeneratorOutput:
    triplets: list[list[str]]


@dataclass
class FactCheckerOutput:
    fact_check_prediction_binary: dict[str, bool]


@dataclass
class HallucinationDataGeneratorOutput:
    generated_hlcntn_answer: str
    generated_non_hlcntn_answer: str
    hlcntn_part: str


@dataclass
class DirectTextMatchOutput:
    input_triplets: list[list[str]]
    reference_triplets: list[list[str]]
    fact_check_prediction_binary: dict[str, bool]
