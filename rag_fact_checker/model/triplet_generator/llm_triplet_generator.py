import logging
from typing import Dict, List

from langchain_core.messages import BaseMessage

from rag_fact_checker.data import Config, TripletGeneratorOutput
from rag_fact_checker.model.triplet_generator import (
    TripletGenerator,
)
from rag_fact_checker.pipeline import PipelineLLM, PipelinePrompt


class LLMTripletGenerator(TripletGenerator, PipelineLLM, PipelinePrompt):
    """
    LLMTripletGenerator is a class that generates triplets from input data using a language model.

    Methods
    -------
    __init__(self, config : Config, logger: logging.Logger)
        Initializes the LLMTripletGenerator with the given configuration and logger.

    forward(self, input_text: str) -> TripletGeneratorOutput
        Generates triplets from the input data.

    default_triplet(self) -> List[str]
        Returns the default triplet.

    get_model_prompt(self, text_input:str, **kwargs) -> List[BaseMessage]
        Creates a prompt for triplet generation using the provided text input or generated answer.

    triplet_generation_input_formatter(self, text_input:str) -> Dict[str, str]
        Formats the input text for triplet generation.

    parse_triplet_generation_output(self, triplet_generation_output:str) -> List[List[str]]
        Parses the output text to extract triplets.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        TripletGenerator.__init__(self, config, logger)
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    def forward(self, input_text: str) -> TripletGeneratorOutput:
        """
        Processes the input data to generate triplets using a model.

        Args:
            input_text (str): The input text data from which triplets are to be generated.

        Returns:
            TripletGeneratorOutput: TripletGeneratorOutput which has a list of triplets generated from the input data.
        """
        triplet_generation_prompt = self.get_model_prompt(input_text)
        triplet_generation_output = self.model.invoke(triplet_generation_prompt).content

        if self.config.experiment_setup.log_prompts:
            self.logger.debug(triplet_generation_prompt)

        return TripletGeneratorOutput(
            triplets=self.parse_triplet_generation_output(triplet_generation_output)
        )

    @property
    def default_triplet(self) -> List[str]:
        """
        Generates a default triplet.
        Returns:
            list: A list containing three empty strings.
        """
        return ["", "", ""]

    def get_model_prompt(self, input_text: str, **kwargs) -> List[BaseMessage]:
        """
        Create a prompt for triplet generation using the provided text input.
        Args:
            input_text (Optional[str]): The input text for generating the prompt.
            **kwargs: Additional keyword arguments. Must include 'generated_answer' if text_input is not provided.

        Returns:
            str: The generated prompt for triplet generation.

        Raises:
            AssertionError: If neither input_text nor 'generated_answer' in kwargs is provided.
        """
        if input_text == None:
            assert (
                "generated_answer" in kwargs
            ), "one of input_text input or generated_answer should be provided"
            input_text = kwargs["generated_answer"]
        return self.message_list_template["triplet_generation"].invoke(
            input=self.triplet_generation_input_formatter(input_text)
        )

    def triplet_generation_input_formatter(self, input_text: str) -> Dict[str, str]:
        """
        Formats the input text for triplet generation.

        Args:
            input_text (str): The input text to be formatted.

        Returns:
            dict: A dictionary containing the formatted input text with the key 'input_text'.
        """
        return {"input_text": input_text}

    def parse_triplet_generation_output(
        self, triplet_generation_model_output: str
    ) -> List[List[str]]:
        """
        Parse output text to triplets.
        Args:
            triplet_generation_model_output (str): The output text containing triplets separated by newlines.

        Returns:
            list: A list of triplets parsed from the output text. If parsing fails or any triplet does not contain exactly three elements, returns the default triplet.

        """
        try:
            result = eval(self.preprocess_output(triplet_generation_model_output))
            for triplet in result:
                if len(triplet) != 3:
                    self.logger.warning(
                        "Some triplet failed in generation : %s", str(triplet)
                    )
            result = [
                triplet if len(triplet) == 3 else self.default_triplet
                for triplet in result
            ]
        except Exception as e:  # todo: how to handle this exception?
            self.logger.warning("Error parsing triplet generation output. : %s", str(e))
            try:
                result = eval(self.preprocess_output(triplet_generation_model_output))
                result = self.default_triplet * len(result)
            except Exception as e:

                result = self.default_triplet * 1
            self.logger.debug("Error occured in : %s", triplet_generation_model_output)
            self.logger.debug("So we used : %s", str(result))
        return result

    def preprocess_output(self, output: str) -> str:
        """
        Preprocesses the given output string by performing a series of string replacements.

        Args:
            output (str): The output string to preprocess.

        Returns:
            str: The preprocessed output string with specific characters and patterns replaced.
        """
        example = (
            output.strip()
            .replace("{", "")
            .replace("}", "")
            .replace("]]]", "]]")
            .replace("]].", "]]")
        )
        return example
