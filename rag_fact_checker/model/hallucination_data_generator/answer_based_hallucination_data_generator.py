import logging
from enum import Enum

from rag_fact_checker.data import Config, HallucinationDataGeneratorOutput
from rag_fact_checker.model.hallucination_data_generator.hallucination_data_generator import (
    HallucinationDataGenerator,
)
from rag_fact_checker.pipeline.simple_batch_processor import (
    SimpleBatchProcessingMixin,
    SimpleBatchResult,
)


class ErrorType(Enum):
    """Types of errors that can be injected into correct answers."""

    FACTUAL = "factual"  # Change facts/entities
    TEMPORAL = "temporal"  # Change dates, time periods
    NUMERICAL = "numerical"  # Change numbers, quantities
    RELATIONAL = "relational"  # Change relationships between entities
    CONTEXTUAL = "contextual"  # Add unrelated context
    OMISSION = "omission"  # Remove important details


class AnswerBasedHallucinationDataGenerator(
    HallucinationDataGenerator, SimpleBatchProcessingMixin
):
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

        # Define JSON schema for answer-based hallucination output (optimized - no redundant original_answer)
        answer_hallucination_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_hallucination_response",
                "schema": {
                    "type": "object",
                    "properties": {
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

        hallucinated_answer, error_details = (
            self.parse_answer_based_hallucination_output(hallucination_output)
        )

        return HallucinationDataGeneratorOutput(
            generated_non_hlcntn_answer=correct_answer,  # Use input directly - no need to parse redundant data
            generated_hlcntn_answer=hallucinated_answer,
            hlcntn_part=error_details,
        )

    async def generate_answer_based_hallucination_async(
        self,
        correct_answer: str,
        question: str,
        error_types: list[ErrorType] | None = None,
        intensity: float = 0.3,
    ) -> HallucinationDataGeneratorOutput:
        """
        Generate hallucinated data by injecting specific errors into a correct answer (async version).

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

        # Define JSON schema for answer-based hallucination output (optimized - no redundant original_answer)
        answer_hallucination_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_hallucination_response",
                "schema": {
                    "type": "object",
                    "properties": {
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
                        "hallucinated_answer",
                        "injected_errors",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        # Use async client for true async operation
        response = await self.async_model.chat.completions.create(
            model=self.config.model.llm.generator_model,
            messages=hallucination_prompt,
            temperature=self.config.model.llm.temperature,
            response_format=answer_hallucination_schema,
        )

        hallucination_output = response.choices[0].message.content

        if self.config.experiment_setup.log_prompts:
            self.logger.debug(hallucination_prompt)

        hallucinated_answer, error_details = (
            self.parse_answer_based_hallucination_output(hallucination_output)
        )

        return HallucinationDataGeneratorOutput(
            generated_non_hlcntn_answer=correct_answer,  # Use input directly - no need to parse redundant data
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
    ) -> tuple[str, list[str]]:
        """
        Parse JSON output from answer-based hallucination generation.

        Args:
            hallucination_output (str): JSON output from the model

        Returns:
            tuple: (hallucinated_answer, hallucinated_parts_list)
        """
        import json

        try:
            data = json.loads(hallucination_output.strip())

            hallucinated_answer = data.get("hallucinated_answer", "").strip()

            # Extract only the modified text (hallucinated parts) as list of strings
            injected_errors = data.get("injected_errors", [])
            hallucinated_parts = []

            for error in injected_errors:
                modified_text = error.get("modified_text", "").strip()
                if modified_text:
                    hallucinated_parts.append(modified_text)

            error_details_string = hallucinated_parts

            return hallucinated_answer, error_details_string

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(
                f"Error parsing answer-based hallucination output: {str(e)}"
            )
            self.logger.debug(f"Raw output: {hallucination_output}")
            return "", []
        except Exception as e:
            self.logger.warning(
                f"Unexpected error parsing answer-based hallucination: {str(e)}"
            )
            self.logger.debug(f"Raw output: {hallucination_output}")
            return "", []

    # Batch processing methods
    def generate_answer_based_hallucination_batch(
        self,
        correct_answers: list[str],
        questions: list[str],
        error_types_list: list[list[ErrorType]] | None = None,
        intensities: list[float] | None = None,
    ) -> SimpleBatchResult[HallucinationDataGeneratorOutput]:
        """
        Generate hallucinated data for multiple correct answer and question pairs concurrently.

        Args:
            correct_answers: List of correct answers to introduce errors into
            questions: List of questions for context
            error_types_list: List of error type lists for each answer. If None, defaults to
                [FACTUAL, TEMPORAL, NUMERICAL] for all answers
            intensities: List of intensity values for each answer. If None, defaults to 0.3 for all

        Returns:
            SimpleBatchResult containing HallucinationDataGeneratorOutput for each successful generation
        """
        batch_size = len(correct_answers)

        if len(questions) != batch_size:
            raise ValueError("Correct answers and questions batch sizes must match")

        # Handle defaults for error_types_list
        if error_types_list is None:
            default_error_types = [
                ErrorType.FACTUAL,
                ErrorType.TEMPORAL,
                ErrorType.NUMERICAL,
            ]
            error_types_list = [default_error_types] * batch_size
        elif len(error_types_list) != batch_size:
            raise ValueError("Error types list and batch size must match")

        # Handle defaults for intensities
        if intensities is None:
            intensities = [0.3] * batch_size
        elif len(intensities) != batch_size:
            raise ValueError("Intensities list and batch size must match")

        # Create tuples for processing
        generation_tasks = list(
            zip(correct_answers, questions, error_types_list, intensities)
        )

        def process_single_task(task_tuple):
            correct_answer, question, error_types, intensity = task_tuple
            return self.generate_answer_based_hallucination(
                correct_answer=correct_answer,
                question=question,
                error_types=error_types,
                intensity=intensity,
            )

        return self.process_items_concurrently(
            generation_tasks,
            process_single_task,
            "answer_based_hallucination_generation_tasks",
        )

    async def generate_answer_based_hallucination_batch_async(
        self,
        correct_answers: list[str],
        questions: list[str],
        error_types_list: list[list[ErrorType]] | None = None,
        intensities: list[float] | None = None,
    ) -> SimpleBatchResult[HallucinationDataGeneratorOutput]:
        """
        Generate hallucinated data for multiple correct answer and question pairs concurrently with TRUE async support.

        Args:
            correct_answers: List of correct answers to introduce errors into
            questions: List of questions for context
            error_types_list: List of error type lists for each answer. If None, defaults to
                [FACTUAL, TEMPORAL, NUMERICAL] for all answers
            intensities: List of intensity values for each answer. If None, defaults to 0.3 for all

        Returns:
            SimpleBatchResult containing HallucinationDataGeneratorOutput for each successful generation
        """
        import asyncio
        import time

        batch_size = len(correct_answers)

        if len(questions) != batch_size:
            raise ValueError("Correct answers and questions batch sizes must match")

        # Handle defaults for error_types_list
        if error_types_list is None:
            default_error_types = [
                ErrorType.FACTUAL,
                ErrorType.TEMPORAL,
                ErrorType.NUMERICAL,
            ]
            error_types_list = [default_error_types] * batch_size
        elif len(error_types_list) != batch_size:
            raise ValueError("Error types list and batch size must match")

        # Handle defaults for intensities
        if intensities is None:
            intensities = [0.3] * batch_size
        elif len(intensities) != batch_size:
            raise ValueError("Intensities list and batch size must match")

        # Enhanced logging
        start_time = time.time()
        self.logger.info(
            f"Starting TRUE async batch processing: {batch_size} answer_based_hallucination_generation_tasks"
        )

        # Create true async tasks
        tasks = []
        for i, (correct_answer, question, error_types, intensity) in enumerate(
            zip(correct_answers, questions, error_types_list, intensities)
        ):
            self.logger.debug(
                f"Creating async task {i + 1}/{batch_size}: error_types={[et.value for et in error_types]}, intensity={intensity}"
            )
            task = self.generate_answer_based_hallucination_async(
                correct_answer=correct_answer,
                question=question,
                error_types=error_types,
                intensity=intensity,
            )
            tasks.append(task)

        # Run all tasks concurrently with progress tracking
        results = []
        failed_indices = []
        errors = []

        try:
            # Use asyncio.gather to run all tasks concurrently
            self.logger.info(f"Running {len(tasks)} async tasks concurrently...")
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {i + 1} failed: {str(result)}")
                    failed_indices.append(i)
                    errors.append(result)
                else:
                    results.append(result)
                    if (i + 1) % max(
                        1, len(tasks) // 10
                    ) == 0:  # Log progress every 10%
                        progress_pct = ((i + 1) * 100) // len(tasks)
                        self.logger.info(
                            f"Progress: {i + 1}/{len(tasks)} tasks completed ({progress_pct}%)"
                        )

        except Exception as e:
            self.logger.error(f"Batch async processing failed: {str(e)}")
            raise

        total_time = time.time() - start_time
        successful_count = len(results)
        failed_count = len(failed_indices)

        self.logger.info(
            f"TRUE async batch processing completed in {total_time:.2f}s: "
            f"{successful_count} successful, {failed_count} failed"
        )
        if failed_count > 0:
            self.logger.error(f"Failed task indices: {failed_indices}")

        return SimpleBatchResult(
            results=results,
            failed_indices=failed_indices,
            errors=errors,
            total_time=total_time,
            successful_count=successful_count,
            failed_count=failed_count,
        )
