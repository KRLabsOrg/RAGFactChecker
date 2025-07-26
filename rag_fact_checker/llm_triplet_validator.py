import os

from easydict import EasyDict as edict

from rag_fact_checker.data import (
    Config,
    DirectTextMatchOutput,
    HallucinationDataGeneratorOutput,
    TripletGeneratorOutput,
)
from rag_fact_checker.model import model_name_class_mapping
from rag_fact_checker.pipeline import PipelineBase
from rag_fact_checker.utils import (
    DEFAULT_CONFIG,
    ExperimentLogger,
    override_config,
)


class LLMTripletValidator(PipelineBase):
    """
    A class representing a system pipeline.
    This class does:
      1)integrates Two main components:a triplet generator and a fact checker
      2)This class defines the main process of our system pipeline.
      3)Defines methods needed in above process.
    """

    def __init__(self, input_config: dict = None, openai_api_key: str = None):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        assert openai_api_key is not None, (
            "OpenAI API key is required. Please pass it as input or set it in the environment."
        )

        config = self.load_config(openai_api_key=openai_api_key, config=input_config)

        super().__init__(config)

        logger = ExperimentLogger(
            "",
            log_path="",
            logger_level=config.logger_level,
        )

        self.logger = logger

        self.triplet_generator = model_name_class_mapping["triplet_generator"][
            config.model.triplet_generator.model_name
        ](config, logger)
        self.fact_checker = model_name_class_mapping["fact_checker"][
            config.model.fact_checker.model_name
        ](config, logger)

        self.hlcntn_data_generator = model_name_class_mapping[
            "hallucination_data_generator"
        ][config.model.hallucination_data_generator.model_name](config, logger)

    def load_config(
        self,
        openai_api_key: str,
        config: dict = None,
    ) -> Config:  # config dataclass
        """
        Loads and merges the configuration with default settings and overrides.
        Args:
            config (dict): A dictionary containing configuration settings.
            openai_api_key (str): The API key for OpenAI.
        Returns:
            Config: The merged configuration dictionary.
        """
        if config is None:
            config = {}
        else:
            config = edict(config)
        default_config = edict(DEFAULT_CONFIG.copy())
        default_config.model.llm.api_key = openai_api_key

        config = override_config(default_config, config)

        return config

    def direct_text_match_forward(
        self, input_text: str, reference_text: str
    ) -> DirectTextMatchOutput:
        """
        input:
            - input_text: The text to be checked.
            - reference_text: The reference text to compare against.
        output(DirectTextMatchOutput):
            -  input_triplets: the triplets from the input text
            -  reference_triplets: the triplets from the reference text
            -  fact_check_prediction_binary:the fact checker output
        """

        input_triplets = self.triplet_generator.forward(input_text=input_text).triplets

        reference_triplets = self.triplet_generator.forward(
            input_text=reference_text
        ).triplets

        fact_checker_output = self.fact_checker.forward(
            answer_triplets=input_triplets, reference_triplets=[reference_triplets]
        )

        return DirectTextMatchOutput(
            input_triplets=input_triplets,
            reference_triplets=reference_triplets,
            fact_check_prediction_binary=fact_checker_output.fact_check_prediction_binary,
        )

    def validate_llm_triplets(
        self, input_text: str, reference_text: list
    ) -> DirectTextMatchOutput:
        """
        Perform forward pass for direct text matching.

        Args:
            input_text (str): The text to be checked.
            reference_text (list): The reference text to compare against.

        Returns:
            DirectTextMatchOutput: A dictionary containing the following:
                - "input_triplets" (list): The triplets extracted from the input text.
                - "fact_check_prediction_binary" (dict): The binary prediction result of the fact checker.
                - "false_triplet_index" (int): The index of the triplet predicted as False.
        """
        assert input_text is not None
        assert reference_text is not None

        result = self.direct_text_match_forward(input_text, "\n\n".join(reference_text))

        tabs = "\t\t\t\t\t\t\t\t\t"

        self.logger.info(
            f"\n{tabs} Answer: {input_text}"
            f"\n{tabs} Reference: {reference_text}"
            f"\n{tabs} Input triplets: {result.input_triplets}"
            f"\n{tabs} Reference triplets: {result.reference_triplets}"
            f"\n{tabs} Fact checking output: {result.fact_check_prediction_binary}"
        )
        return result

    def triplet_generation(self, input_text: str) -> TripletGeneratorOutput:
        """
        Perform forward pass for direct text matching.

        Args:
            input_text (str): The text to be checked.

        Returns:
            DirectTextMatchOutput: A dictionary containing the following:
                - "input_triplets" (list): The triplets extracted from the input text.
                - "fact_check_prediction_binary" (dict): The binary prediction result of the fact checker.
                - "false_triplet_index" (int): The index of the triplet predicted as False.
        """
        assert input_text is not None

        result = self.triplet_generator.forward(input_text=input_text)

        tabs = "\t\t\t\t\t\t\t\t\t"

        self.logger.info(
            f"\n{tabs} Answer: {input_text}"
            f"\n{tabs} generated triplets: {result.triplets}"
        )

        return result

    def generate_hlcntn_data(
        self, question: str, reference_text: list[str]
    ) -> HallucinationDataGeneratorOutput:
        """
        Perform forward pass for hallucination data generation.

        Args:
            question (str): The question to be asked.
            reference_text (str): The reference text to be used for hallucination.

        Returns:
            HallucinationDataGeneratorOutput: A dictionary containing the following:
                - "generated_hlcntn_answer" (str): The generated hallucinated answer.
                - "generated_non_hlcntn_answer" (str): The generated non-hallucinated answer.
                - "hlcntn_part" (str): The hallucinated details.
        """
        assert reference_text is not None
        assert question is not None

        result = self.hlcntn_data_generator.generate_hlcntn_data(
            reference_text, question
        )

        tabs = "\t\t\t\t\t\t\t\t\t"

        self.logger.info(
            f"\n{tabs} Reference: {reference_text}"
            f"\n{tabs} Question: {question}"
            f"\n{tabs} Generated hallucinated answer: {result.generated_hlcntn_answer}"
            f"\n{tabs} Generated non-hallucinated answer: {result.generated_non_hlcntn_answer}"
            f"\n{tabs} Hallucination part: {result.hlcntn_part}"
        )

        return result
