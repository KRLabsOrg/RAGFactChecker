import logging
from abc import abstractmethod

from rag_fact_checker.data import Config, HallucinationDataGeneratorOutput
from rag_fact_checker.pipeline import PipelineLLM, PipelinePrompt


class HallucinationDataGenerator(PipelineLLM, PipelinePrompt):
    """
    HallucinationDataGenerator is used to generate data for hallucination detection in language models.

    Attributes:
        config (Config): Configuration dataclass for initializing the class.
        logger (logging.Logger): Logger object for logging.

    Methods:
        __init__(config: config: Config, logger: logging.Logger):
            Initializes the HallucinationDataGenerator with the given configuration and logger.

    Note:
        hallucination dataset looks like this:
        {
            "generated_hlcntn_answer":
        }
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.logger = logger
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    @abstractmethod
    def generate_hlcntn_data(
        self,
        question: str,
        reference_text: str,
    ) -> HallucinationDataGeneratorOutput:
        raise NotImplementedError
