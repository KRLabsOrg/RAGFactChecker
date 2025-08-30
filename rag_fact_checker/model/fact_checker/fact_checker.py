import logging
from abc import abstractmethod

from rag_fact_checker.data import Config, FactCheckerOutput
from rag_fact_checker.pipeline import PipelineBase


class FactChecker(PipelineBase):
    """
    FactChecker is a base(abstract) class for implementing fact-checking pipelines. It inherits from rag_fact_checker.pipelineBase and requires a configuration dictionary during initialization.

    Methods:

    __init__(config: Config, logger: logging.Logger)
        Initializes the FactChecker with the given configuration and logger.

    forward(FactCheckerInput) -> FactCheckerOutput
        Abstract method to generate triplets from the data. Must be implemented by subclasses.

    """

    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__(config)
        self.logger = logger

    @abstractmethod
    def forward(
        self,
        answer_triplets: list[list[str]],
        reference_triplets: list[list[list[str]]],
    ) -> FactCheckerOutput:
        """
        Generate triplets from the data
        """
        raise NotImplementedError

    def flatten_triplets(
        self, triplet_segments: list[list[list[str]]]
    ) -> list[list[str]]:
        """
        Flatten the list of triplets into a single list of strings.
        """
        return [triplet for sublist in triplet_segments for triplet in sublist]

    def merge_segment_outputs(
        self, output_list: list[FactCheckerOutput]
    ) -> FactCheckerOutput:
        if not output_list:
            self.logger.error("Empty fect check output list")

        # check if all dictionaries have the same keys
        keys_list = [set(d.fact_check_prediction_binary.keys()) for d in output_list]
        if not all(keys == keys_list[0] for keys in keys_list):
            self.logger.warning("Not all dictionaries have the same keys.")
            self.logger.debug("Keys: %s", keys_list)
            return {}

        all_keys = set().union(*keys_list)
        merged_fact_check_result = {key: False for key in all_keys}

        # merge the dictionaries
        for d in output_list:
            for key, value in d.fact_check_prediction_binary.items():
                if value:  # if any of the dictionaries has the key as True, set the result to True
                    merged_fact_check_result[key] = True

        return FactCheckerOutput(fact_check_prediction_binary=merged_fact_check_result)
