import logging
from abc import abstractmethod

from rag_fact_checker.data import Config, TripletGeneratorOutput
from rag_fact_checker.pipeline import PipelineBase


class TripletGenerator(PipelineBase):
    """
    TripletGenerator is a base class for generating triplets from input data. It inherits from rag_fact_checker.pipelineBase and requires a configuration dictionary during initialization.

    Methods:
        __init__(config: Config, logger: logging.Logger):
            Initializes the TripletGenerator with the given configuration and logger.

        forward(input_text: str) -> TripletGeneratorOutput:
            Abstract method that must be implemented by subclasses to generate triplets from the input data. Raises NotImplementedError if not overridden.

    Properties:
        input_output_format:
            Returns a dictionary specifying the expected input and output format for the triplet generation process.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__(config)
        self.logger = logger

    @abstractmethod
    def forward(self, input_text: str) -> TripletGeneratorOutput:
        """
        Generate triplets from the data
        """
        raise NotImplementedError
