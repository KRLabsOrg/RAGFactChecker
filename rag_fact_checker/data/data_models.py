from dataclasses import dataclass, field


@dataclass
class ExperimentSetupConfig:
    system_retry: int = 2
    log_prompts: bool = False


@dataclass
class AnswerGeneratorConfig:
    model_name: str = "llm"
    num_shot: int = 2


@dataclass
class TripletGeneratorModelParams:
    openie_affinity_probability_cap: float = 0.6


@dataclass
class TripletGeneratorConfig:
    model_name: str = "llm_n_shot"
    model_params: TripletGeneratorModelParams = field(
        default_factory=TripletGeneratorModelParams
    )
    num_shot: int = 3


@dataclass
class FactCheckerConfig:
    model_name: str = "llm"
    split_reference_triplets: bool = True
    max_reference_triplet_length: int = 100
    num_shot: int = 2
    inquiry_mode: bool = True


@dataclass
class LLMConfig:
    generator_model: str = "gpt-4o"
    request_max_try: int = 1
    temperature: float = 0.0
    api_key: str | None = field(
        default_factory=lambda: __import__("os").getenv("OPENAI_API_KEY")
    )
    base_url: str | None = field(
        default_factory=lambda: __import__("os").getenv("OPENAI_BASE_URL")
    )


@dataclass
class SimpleBatchConfig:
    """Configuration for simple batch processing."""

    max_workers: int = 5  # Number of concurrent threads
    max_retries: int = 3  # Maximum retries for failed items
    retry_delay: float = 1.0  # Delay between retries in seconds
    timeout: float | None = None  # Timeout per individual call


@dataclass
class ModelConfig:
    answer_generator: AnswerGeneratorConfig = field(
        default_factory=AnswerGeneratorConfig
    )
    triplet_generator: TripletGeneratorConfig = field(
        default_factory=TripletGeneratorConfig
    )
    fact_checker: FactCheckerConfig = field(default_factory=FactCheckerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    hallucination_data_generator: AnswerGeneratorConfig = field(
        default_factory=AnswerGeneratorConfig
    )


@dataclass
class PathDataConfig:
    base: str = "rag_fact_checker/data/"
    demo: str = "demonstrations"


def _get_default_prompt_path():
    """Get the correct path to prompt_bank.json."""
    import pathlib

    # Get the rag_fact_checker package directory
    rag_fact_checker_dir = pathlib.Path(__file__).parent.parent
    prompt_path = rag_fact_checker_dir / "prompt_bank.json"
    return str(prompt_path)


@dataclass
class PathConfig:
    data: PathDataConfig = field(default_factory=PathDataConfig)
    prompts: str = field(default_factory=_get_default_prompt_path)


@dataclass
class Config:
    experiment_setup: ExperimentSetupConfig = field(
        default_factory=ExperimentSetupConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    path: PathConfig = field(default_factory=PathConfig)
    logger_level: str | None = None
    simple_batch_config: SimpleBatchConfig = field(default_factory=SimpleBatchConfig)


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
    hlcntn_part: list[str]


@dataclass
class DirectTextMatchOutput:
    input_triplets: list[list[str]]
    reference_triplets: list[list[str]]
    fact_check_prediction_binary: dict[str, bool]
