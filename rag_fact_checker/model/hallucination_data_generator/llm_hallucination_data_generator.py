import logging

from rag_fact_checker.data import Config, HallucinationDataGeneratorOutput
from rag_fact_checker.model.hallucination_data_generator.hallucination_data_generator import (
    HallucinationDataGenerator,
)
from rag_fact_checker.pipeline.simple_batch_processor import (
    SimpleBatchProcessingMixin,
    SimpleBatchResult,
)


class LLMHallucinationDataGenerator(
    HallucinationDataGenerator, SimpleBatchProcessingMixin
):
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
    ) -> list[dict[str, str]]:
        """
        Generates a model prompt for hallucinated data generation.

        Args:
            reference_documents (list of str): A list of reference documents to be used in the prompt.
            question (str): The question to be answered by the model.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, str]]: The generated model prompt.
        """
        template_names = self.message_list_template["hallucinated_data_generation_test"]
        return self.create_messages(
            template_names,
            **self.hlcntn_prompt_input_formatter(reference_documents, question),
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
        # Define JSON schema for structured outputs
        hallucination_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "hallucination_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "non_hallucinated_answer": {
                            "type": "string",
                            "description": "Comprehensive, evidence-based answer using only reference documents",
                        },
                        "hallucinated_answer": {
                            "type": "string",
                            "description": "Same as non_hallucinated_answer but with subtle fictional details added",
                        },
                        "hallucinated_details": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of each hallucinated fact as separate elements",
                        },
                    },
                    "required": [
                        "non_hallucinated_answer",
                        "hallucinated_answer",
                        "hallucinated_details",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = self.model.chat.completions.create(
            model=self.config.model.llm.generator_model,
            messages=hlcntn_generation_prompt,
            temperature=self.config.model.llm.temperature,
            response_format=hallucination_schema,
        )
        hlcntn_data_generation_output = response.choices[0].message.content

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

    async def generate_hlcntn_data_async(
        self, reference_text: str, question: str
    ) -> HallucinationDataGeneratorOutput:
        """
        Perform forward pass for hallucination data generation (async version).

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
        # Define JSON schema for structured outputs
        hallucination_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "hallucination_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "non_hallucinated_answer": {
                            "type": "string",
                            "description": "Comprehensive, evidence-based answer using only reference documents",
                        },
                        "hallucinated_answer": {
                            "type": "string",
                            "description": "Same as non_hallucinated_answer but with subtle fictional details added",
                        },
                        "hallucinated_details": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of each hallucinated fact as separate elements",
                        },
                    },
                    "required": [
                        "non_hallucinated_answer",
                        "hallucinated_answer",
                        "hallucinated_details",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        # Use async client for true async operation
        response = await self.async_model.chat.completions.create(
            model=self.config.model.llm.generator_model,
            messages=hlcntn_generation_prompt,
            temperature=self.config.model.llm.temperature,
            response_format=hallucination_schema,
        )
        hlcntn_data_generation_output = response.choices[0].message.content

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
    ) -> tuple[str, str, list[str]]:
        """
        Parses the hallucination data generation JSON output and extracts the components.

        Args:
            hlcntn_data_generation_output (str): The JSON output string from the hallucination data generation process.

        Returns:
            tuple: A tuple containing:
                - non_hlcntn_answer (str): The non-hallucinated answer extracted from the output.
                - hlcntn_answer (str): The hallucinated answer extracted from the output.
                - hlcntn_part (list[str]): The hallucinated details extracted from the output as a list of strings.
        """
        import json

        try:
            # Parse JSON output
            data = json.loads(hlcntn_data_generation_output.strip())

            non_hlcntn_answer = data.get("non_hallucinated_answer", "").strip()
            hlcntn_answer = data.get("hallucinated_answer", "").strip()

            # Extract hallucinated details as list of strings
            hlcntn_part = data.get("hallucinated_details", [])

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Error parsing JSON hallucination output: {str(e)}")
            self.logger.debug(
                f"Raw hallucination output: {hlcntn_data_generation_output}"
            )
            # Fallback to empty values
            non_hlcntn_answer, hlcntn_answer, hlcntn_part = "", "", []
        except Exception as e:
            self.logger.warning(
                f"Unexpected error parsing hallucination output: {str(e)}"
            )
            self.logger.debug(
                f"Raw hallucination output: {hlcntn_data_generation_output}"
            )
            non_hlcntn_answer, hlcntn_answer, hlcntn_part = "", "", []

        return non_hlcntn_answer, hlcntn_answer, hlcntn_part

    # Batch processing methods
    def generate_hlcntn_data_batch(
        self, reference_texts: list[str], questions: list[str]
    ) -> SimpleBatchResult[HallucinationDataGeneratorOutput]:
        """
        Generate hallucination data for multiple reference text and question pairs concurrently.

        Args:
            reference_texts: List of reference texts
            questions: List of questions

        Returns:
            SimpleBatchResult containing HallucinationDataGeneratorOutput for each successful generation
        """
        if len(reference_texts) != len(questions):
            raise ValueError("Reference texts and questions batch sizes must match")

        # Create tuples for processing
        generation_tasks = list(zip(reference_texts, questions))

        def process_single_task(task_tuple):
            reference_text, question = task_tuple
            return self.generate_hlcntn_data(reference_text, question)

        return self.process_items_concurrently(
            generation_tasks, process_single_task, "hallucination_generation_tasks"
        )

    async def generate_hlcntn_data_batch_async(
        self, reference_texts: list[str], questions: list[str]
    ) -> SimpleBatchResult[HallucinationDataGeneratorOutput]:
        """
        Generate hallucination data for multiple reference text and question pairs concurrently with TRUE async support.

        Args:
            reference_texts: List of reference texts
            questions: List of questions

        Returns:
            SimpleBatchResult containing HallucinationDataGeneratorOutput for each successful generation
        """
        import asyncio
        import time

        if len(reference_texts) != len(questions):
            raise ValueError("Reference texts and questions batch sizes must match")

        batch_size = len(reference_texts)

        # Enhanced logging
        start_time = time.time()
        self.logger.info(
            f"Starting TRUE async batch processing: {batch_size} llm_hallucination_generation_tasks"
        )

        # Create true async tasks
        tasks = []
        for i, (reference_text, question) in enumerate(zip(reference_texts, questions)):
            self.logger.debug(
                f"Creating async task {i + 1}/{batch_size} for question: '{question[:50]}...'"
            )
            task = self.generate_hlcntn_data_async(reference_text, question)
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
