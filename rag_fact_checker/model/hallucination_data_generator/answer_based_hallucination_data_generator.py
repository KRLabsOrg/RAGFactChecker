import logging
from enum import Enum

from rag_fact_checker.data import Config, HallucinationDataGeneratorOutput
from rag_fact_checker.model.hallucination_data_generator.hallucination_data_generator import (
    HallucinationDataGenerator,
)


class ErrorType(Enum):
    """Types of errors that can be injected into correct answers."""

    FACTUAL = "factual"  # Change facts/entities
    TEMPORAL = "temporal"  # Change dates, time periods
    NUMERICAL = "numerical"  # Change numbers, quantities
    RELATIONAL = "relational"  # Change relationships between entities
    CONTEXTUAL = "contextual"  # Add unrelated context
    OMISSION = "omission"  # Remove important details


class AnswerBasedHallucinationDataGenerator(HallucinationDataGenerator):
    """
    Generates hallucinated data by taking a correct answer and injecting specific types of errors.

    This addresses the limitation where users couldn't systematically introduce controlled
    hallucinations into known correct answers.

    Methods:
    --------
    generate_answer_based_hallucination(correct_answer, question, error_types, intensity) -> HallucinationDataGeneratorOutput
        Takes a correct answer and injects specified types of errors to create hallucinated version.

    get_answer_based_model_prompt(correct_answer, question, error_types, intensity) -> List[Dict[str, str]]
        Creates prompt for answer-based hallucination generation.

    answer_based_input_formatter(correct_answer, question, error_types, intensity) -> Dict[str, str]
        Formats input for answer-based hallucination prompt.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__(config, logger)

    def generate_answer_based_hallucination(
        self,
        correct_answer: str,
        question: str,
        error_types: list[ErrorType] | None = None,
        intensity: float = 0.3,
    ) -> HallucinationDataGeneratorOutput:
        """
        Generate hallucinated data by injecting specific errors into a correct answer.

        Args:
            correct_answer (str): The known correct answer to introduce errors into
            question (str): The original question for context
            error_types (List[ErrorType], optional): Types of errors to inject.
                Defaults to [FACTUAL, TEMPORAL, NUMERICAL]
            intensity (float): Error intensity from 0.1 (subtle) to 1.0 (obvious).
                Defaults to 0.3 (moderate)

        Returns:
            HallucinationDataGeneratorOutput: Contains original correct answer,
                hallucinated version, and details of injected errors
        """
        if error_types is None:
            error_types = [ErrorType.FACTUAL, ErrorType.TEMPORAL, ErrorType.NUMERICAL]

        # Validate intensity
        if not 0.1 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.1 and 1.0")

        hallucination_prompt = self.get_answer_based_model_prompt(
            correct_answer=correct_answer,
            question=question,
            error_types=error_types,
            intensity=intensity,
        )

        # Define JSON schema for answer-based hallucination output
        answer_hallucination_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_hallucination_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "original_answer": {
                            "type": "string",
                            "description": "The original correct answer provided as input",
                        },
                        "hallucinated_answer": {
                            "type": "string",
                            "description": "The answer with injected errors of specified types and intensity",
                        },
                        "injected_errors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "error_type": {"type": "string"},
                                    "original_text": {"type": "string"},
                                    "modified_text": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": [
                                    "error_type",
                                    "original_text",
                                    "modified_text",
                                    "description",
                                ],
                                "additionalProperties": False,
                            },
                            "description": "List of specific errors injected with details",
                        },
                    },
                    "required": [
                        "original_answer",
                        "hallucinated_answer",
                        "injected_errors",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = self.model.chat.completions.create(
            model=self.config.model.llm.generator_model,
            messages=hallucination_prompt,
            temperature=self.config.model.llm.temperature,
            response_format=answer_hallucination_schema,
        )

        hallucination_output = response.choices[0].message.content

        if self.config.experiment_setup.log_prompts:
            self.logger.debug(hallucination_prompt)

        original_answer, hallucinated_answer, error_details = (
            self.parse_answer_based_hallucination_output(hallucination_output)
        )

        return HallucinationDataGeneratorOutput(
            generated_non_hlcntn_answer=original_answer,
            generated_hlcntn_answer=hallucinated_answer,
            hlcntn_part=error_details,
        )

    def get_answer_based_model_prompt(
        self,
        correct_answer: str,
        question: str,
        error_types: list[ErrorType],
        intensity: float,
    ) -> list[dict[str, str]]:
        """
        Generate model prompt for answer-based hallucination.

        Args:
            correct_answer (str): The correct answer to modify
            question (str): The original question for context
            error_types (List[ErrorType]): Types of errors to inject
            intensity (float): Error intensity level

        Returns:
            List[Dict[str, str]]: Formatted prompt for the model
        """
        template_names = self.message_list_template[
            "answer_based_hallucination_generation"
        ]
        return self.create_messages(
            template_names,
            **self.answer_based_input_formatter(
                correct_answer, question, error_types, intensity
            ),
        )

    def answer_based_input_formatter(
        self,
        correct_answer: str,
        question: str,
        error_types: list[ErrorType],
        intensity: float,
    ) -> dict[str, str]:
        """
        Format input for answer-based hallucination prompt.

        Args:
            correct_answer (str): The correct answer to modify
            question (str): The original question
            error_types (List[ErrorType]): Types of errors to inject
            intensity (float): Error intensity level

        Returns:
            Dict[str, str]: Formatted input dictionary
        """
        error_descriptions = {
            ErrorType.FACTUAL: "Change specific facts, entities, or claims",
            ErrorType.TEMPORAL: "Modify dates, time periods, or temporal relationships",
            ErrorType.NUMERICAL: "Alter numbers, quantities, percentages, or measurements",
            ErrorType.RELATIONAL: "Change relationships between entities or concepts",
            ErrorType.CONTEXTUAL: "Add unrelated or incorrect contextual information",
            ErrorType.OMISSION: "Remove important details or qualifications",
        }

        error_instructions = "\n".join(
            [
                f"- {error_type.value.title()}: {error_descriptions[error_type]}"
                for error_type in error_types
            ]
        )

        intensity_description = self._get_intensity_description(intensity)

        return {
            "correct_answer": correct_answer,
            "question": question,
            "error_types": error_instructions,
            "intensity": intensity_description,
            "intensity_value": str(intensity),
        }

    def _get_intensity_description(self, intensity: float) -> str:
        """Get human-readable description of intensity level."""
        if intensity <= 0.2:
            return "Very subtle errors that are hard to detect"
        elif intensity <= 0.4:
            return "Moderate errors that are noticeable but plausible"
        elif intensity <= 0.6:
            return "Clear errors that are obviously incorrect"
        elif intensity <= 0.8:
            return "Strong errors that significantly change meaning"
        else:
            return "Extreme errors that completely contradict the original"

    def parse_answer_based_hallucination_output(
        self, hallucination_output: str
    ) -> tuple[str, str, list[str]]:
        """
        Parse JSON output from answer-based hallucination generation.

        Args:
            hallucination_output (str): JSON output from the model

        Returns:
            tuple: (original_answer, hallucinated_answer, hallucinated_parts_list)
        """
        import json

        try:
            data = json.loads(hallucination_output.strip())

            original_answer = data.get("original_answer", "").strip()
            hallucinated_answer = data.get("hallucinated_answer", "").strip()

            # Extract only the modified text (hallucinated parts) as list of strings
            injected_errors = data.get("injected_errors", [])
            hallucinated_parts = []

            for error in injected_errors:
                modified_text = error.get('modified_text', '').strip()
                if modified_text:
                    hallucinated_parts.append(modified_text)

            error_details_string = hallucinated_parts

            return original_answer, hallucinated_answer, error_details_string

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(
                f"Error parsing answer-based hallucination output: {str(e)}"
            )
            self.logger.debug(f"Raw output: {hallucination_output}")
            return "", "", []
        except Exception as e:
            self.logger.warning(
                f"Unexpected error parsing answer-based hallucination: {str(e)}"
            )
            self.logger.debug(f"Raw output: {hallucination_output}")
            return "", "", []
