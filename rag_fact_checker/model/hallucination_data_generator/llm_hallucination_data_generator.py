import logging
from typing import List

from langchain_core.messages import BaseMessage

from rag_fact_checker.data import HallucinationDataGeneratorOutput, Config
from rag_fact_checker.model.hallucination_data_generator import (
    HallucinationDataGenerator,
)


class LLMHallucinationDataGenerator(HallucinationDataGenerator):
    """
    This class contains hallucination pipelines to generate hallucination data.

    Methods:
    __init__(self, config: Config, logger: logging.Logger)
        Initializes the data generator with the given configuration and logger.

    hlcntn_directions(self)
        Property that returns a list of directions for generating hallucination data.

    get_model_prompt(self, reference_documents, question, **kwargs)
        Generates a model prompt for hallucinated data generation based on reference documents and a question.

    hlcntn_prompt_input_formatter(self, reference_documents, question)
        Formats the input for the hallucination prompt.

    generate_hlcntn_data(self, reference_text: str, question: str)
        Generates hallucinated data from reference_text, question.


    parse_hlcntn_data_generation_output(self, hlcntn_data_generation_output)
        Parses the output of the hallucinated data generation to extract non-hallucinated and hallucinated answers, and the hallucinated part.

    Note:
        hallucination dataset looks like this:
        {
            "generated_hlcntn_answer"
            "generated_non_hlcntn_answer"
            "hlcntn_part"
        }
    """

    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__(config, logger)

    def get_model_prompt(
        self, reference_documents: list, question: str, **kwargs
    ) -> List[BaseMessage]:
        """
        Generates a model prompt for hallucinated data generation.

        Args:
            reference_documents (list of str): A list of reference documents to be used in the prompt.
            question (str): The question to be answered by the model.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated model prompt.
        """
        return self.message_list_template["hallucinated_data_generation_test"].invoke(
            input=self.hlcntn_prompt_input_formatter(reference_documents, question)
        )

    def hlcntn_prompt_input_formatter(
        self, reference_documents: list, question: str
    ) -> dict[str, str]:
        """
        Formats the input for hallucination prompt.

        Args:
            reference_documents (list of str): A list of reference documents.
            question (str): The question to be asked.

        Returns:
            dict: A dictionary containing formatted directions, reference documents, and the question.
        """
        return {
            "reference_documents": "\n-".join(reference_documents),
            "question": question,
        }

    def generate_hlcntn_data(
        self, reference_text: str, question: str
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

        hlcntn_generation_prompt = self.get_model_prompt(
            reference_documents=reference_text,
            question=question,
        )
        hlcntn_data_generation_output = self.model.invoke(
            hlcntn_generation_prompt
        ).content

        generated_non_hlcntn_answer, generated_hlcntn_answer, hlcntn_part = (
            self.parse_hlcntn_data_generation_output(hlcntn_data_generation_output)
        )

        return HallucinationDataGeneratorOutput(
            **{
                "generated_non_hlcntn_answer": generated_non_hlcntn_answer,
                "generated_hlcntn_answer": generated_hlcntn_answer,
                "hlcntn_part": hlcntn_part,
            }
        )

    def parse_hlcntn_data_generation_output(
        self, hlcntn_data_generation_output: str
    ) -> str:
        """
        Parses the hallucination data generation output and extracts the non-hallucinated answer,
        hallucinated answer, and hallucinated details.

        Args:
            hlcntn_data_generation_output (str): The output string from the hallucination data generation process.

        Returns:
            tuple: A tuple containing:
                - non_hlcntn_answer (str): The non-hallucinated answer extracted from the output.
                - hlcntn_answer (str): The hallucinated answer extracted from the output.
                - hlcntn_part (str): The hallucinated details extracted from the output.
        """
        try:
            answer_part = hlcntn_data_generation_output.split("Hallucinated Details:")[
                0
            ]
            hlcntn_part = hlcntn_data_generation_output.split("Hallucinated Details:")[
                1
            ]
            hlcntn_answer = answer_part.split("Hallucinated Answer:\n")[2].replace(
                "*", ""
            )
            non_hlcntn_answer = answer_part.split("Hallucinated Answer:\n")[1].replace(
                "Non-Hallucinated Answer:\n", ""
            )
        except Exception as e:
            self.logger.warning(
                f"Error occurred while parsing hallucination data generation output: {str(e)}"
            )
            self.logger.debug(
                f"hallucination data generation output: , {hlcntn_data_generation_output}"
            )
            non_hlcntn_answer, hlcntn_answer, hlcntn_part = "", "", ""
        return non_hlcntn_answer, hlcntn_answer, hlcntn_part
