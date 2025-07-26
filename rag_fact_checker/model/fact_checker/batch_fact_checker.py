import asyncio
import json
import logging

from rag_fact_checker.data import Config, FactCheckerOutput
from rag_fact_checker.model.fact_checker import FactChecker
from rag_fact_checker.pipeline import PipelineLLM, PipelinePrompt
from rag_fact_checker.pipeline.batch_processor import BatchProcessingMixin, BatchResult


class BatchFactChecker(FactChecker, PipelineLLM, PipelinePrompt, BatchProcessingMixin):
    """
    Batch-enabled fact checker that can process multiple triplet comparisons efficiently.

    Features:
    - Batch processing of multiple fact checking tasks
    - Async and sync batch processing
    - Configurable batch sizes and concurrency
    - Automatic retry and error handling
    - Structured JSON output for reliable parsing
    """

    def __init__(self, config: Config, logger: logging.Logger):
        FactChecker.__init__(self, config, logger)
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    def forward_batch_sync(
        self,
        answer_triplets_batch: list[list[list[str]]],
        reference_triplets_batch: list[list[list[str]]],
    ) -> BatchResult[FactCheckerOutput]:
        """
        Process multiple fact checking tasks synchronously in batches.

        Args:
            answer_triplets_batch: List of answer triplet sets to check
            reference_triplets_batch: List of reference triplet sets to compare against

        Returns:
            BatchResult containing FactCheckerOutput for each successful comparison
        """
        if len(answer_triplets_batch) != len(reference_triplets_batch):
            raise ValueError("Answer and reference batch sizes must match")

        # Combine into single items for batch processing
        fact_check_items = list(zip(answer_triplets_batch, reference_triplets_batch))

        return self.process_items_in_batches_sync(
            fact_check_items, self._process_fact_check_batch, "fact_check_tasks"
        )

    async def forward_batch_async(
        self,
        answer_triplets_batch: list[list[list[str]]],
        reference_triplets_batch: list[list[list[str]]],
    ) -> BatchResult[FactCheckerOutput]:
        """
        Process multiple fact checking tasks asynchronously in batches.

        Args:
            answer_triplets_batch: List of answer triplet sets to check
            reference_triplets_batch: List of reference triplet sets to compare against

        Returns:
            BatchResult containing FactCheckerOutput for each successful comparison
        """
        if len(answer_triplets_batch) != len(reference_triplets_batch):
            raise ValueError("Answer and reference batch sizes must match")

        # Combine into single items for batch processing
        fact_check_items = list(zip(answer_triplets_batch, reference_triplets_batch))

        return await self.process_items_in_batches_async(
            fact_check_items, self._process_fact_check_batch_async, "fact_check_tasks"
        )

    def _process_fact_check_batch(
        self, fact_check_items: list[tuple[list[list[str]], list[list[str]]]]
    ) -> list[FactCheckerOutput]:
        """
        Process a batch of fact checking tasks synchronously.

        Args:
            fact_check_items: Batch of (answer_triplets, reference_triplets) pairs

        Returns:
            List of FactCheckerOutput for each comparison
        """
        # Create batch prompt
        batch_prompt = self._create_batch_fact_check_prompt(fact_check_items)

        # Use structured outputs only for non-inquiry mode
        if self.config.model.fact_checker.inquiry_mode:
            # Inquiry mode requires text-based output for explanations
            response = self.model.chat.completions.create(
                model=self.config.model.llm.generator_model,
                messages=batch_prompt,
                temperature=self.config.model.llm.temperature,
            )

            batch_output = response.choices[0].message.content
            return self._parse_batch_fact_check_inquiry_output(
                batch_output, len(fact_check_items)
            )
        else:
            # Define JSON schema for batch fact checking output
            batch_fact_check_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "batch_fact_check_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "batch_results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task_index": {"type": "integer"},
                                        "results": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "triplet_idx": {"type": "integer"},
                                                    "result": {"type": "boolean"},
                                                },
                                                "required": ["triplet_idx", "result"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["task_index", "results"],
                                    "additionalProperties": False,
                                },
                                "description": "Array of fact check results for each task",
                            }
                        },
                        "required": ["batch_results"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }

            response = self.model.chat.completions.create(
                model=self.config.model.llm.generator_model,
                messages=batch_prompt,
                temperature=self.config.model.llm.temperature,
                response_format=batch_fact_check_schema,
            )

            batch_output = response.choices[0].message.content
            return self._parse_batch_fact_check_output(
                batch_output, len(fact_check_items)
            )

    async def _process_fact_check_batch_async(
        self, fact_check_items: list[tuple[list[list[str]], list[list[str]]]]
    ) -> list[FactCheckerOutput]:
        """
        Process a batch of fact checking tasks asynchronously.

        Args:
            fact_check_items: Batch of (answer_triplets, reference_triplets) pairs

        Returns:
            List of FactCheckerOutput for each comparison
        """
        # For now, wrap sync call in async - could be enhanced with true async OpenAI client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._process_fact_check_batch, fact_check_items
        )

    def _create_batch_fact_check_prompt(
        self, fact_check_items: list[tuple[list[list[str]], list[list[str]]]]
    ) -> list[dict[str, str]]:
        """
        Create a prompt for batch fact checking multiple comparisons.

        Args:
            fact_check_items: List of (answer_triplets, reference_triplets) pairs

        Returns:
            Formatted prompt for batch fact checking
        """
        # Format tasks with indices
        batch_tasks = []
        for task_idx, (answer_triplets, reference_triplets) in enumerate(
            fact_check_items
        ):
            answer_str = "\n-".join(
                [
                    f"{idx}: {str(triplet)}"
                    for idx, triplet in enumerate(answer_triplets)
                ]
            )
            reference_str = "\n-".join(
                [f"{str(triplet)}" for triplet in reference_triplets]
            )

            task_str = f"Task {task_idx}:\nAnswer Triplets:\n{answer_str}\n\nReference Triplets:\n{reference_str}"
            batch_tasks.append(task_str)

        batch_input_text = "\n\n" + "=" * 50 + "\n\n".join(batch_tasks)

        if self.config.model.fact_checker.inquiry_mode:
            template_names = self.message_list_template["batch_fact_check_inquiry"]
        else:
            template_names = self.message_list_template["batch_fact_check"]

        return self.create_messages(
            template_names,
            batch_input_text=batch_input_text,
            num_tasks=str(len(fact_check_items)),
        )

    def _parse_batch_fact_check_output(
        self, batch_output: str, expected_count: int
    ) -> list[FactCheckerOutput]:
        """
        Parse JSON batch fact check output into individual FactCheckerOutput objects.

        Args:
            batch_output: JSON string from the model
            expected_count: Expected number of results

        Returns:
            List of FactCheckerOutput objects
        """
        try:
            data = json.loads(batch_output.strip())
            batch_results = data.get("batch_results", [])

            # Initialize results array with empty predictions
            results = [
                FactCheckerOutput(fact_check_prediction_binary={})
                for _ in range(expected_count)
            ]

            # Fill in actual results
            for result in batch_results:
                if (
                    isinstance(result, dict)
                    and "task_index" in result
                    and "results" in result
                ):
                    task_idx = result["task_index"]
                    if 0 <= task_idx < expected_count:
                        fact_check_results = {}
                        for triplet_result in result["results"]:
                            if (
                                isinstance(triplet_result, dict)
                                and "triplet_idx" in triplet_result
                                and "result" in triplet_result
                            ):
                                fact_check_results[triplet_result["triplet_idx"]] = (
                                    triplet_result["result"]
                                )

                        results[task_idx] = FactCheckerOutput(
                            fact_check_prediction_binary=fact_check_results
                        )
                    else:
                        self.logger.warning(
                            f"Invalid task index in batch result: {task_idx}"
                        )
                else:
                    self.logger.warning(f"Invalid batch result structure: {result}")

            return results

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Error parsing batch fact check output: {str(e)}")
            self.logger.debug(f"Raw batch output: {batch_output}")
            # Return empty predictions for all tasks
            return [
                FactCheckerOutput(fact_check_prediction_binary={})
                for _ in range(expected_count)
            ]
        except Exception as e:
            self.logger.warning(
                f"Unexpected error parsing batch fact check output: {str(e)}"
            )
            self.logger.debug(f"Raw batch output: {batch_output}")
            return [
                FactCheckerOutput(fact_check_prediction_binary={})
                for _ in range(expected_count)
            ]

    def _parse_batch_fact_check_inquiry_output(
        self, batch_output: str, expected_count: int
    ) -> list[FactCheckerOutput]:
        """
        Parse text-based batch inquiry output into FactCheckerOutput objects.

        Args:
            batch_output: Text output from inquiry mode
            expected_count: Expected number of results

        Returns:
            List of FactCheckerOutput objects
        """
        # For inquiry mode, we need to parse the text format
        # This is more complex and would require parsing multiple [FINAL ANSWER] sections
        # For now, return empty results and log warning
        self.logger.warning("Batch inquiry mode parsing not yet implemented")
        return [
            FactCheckerOutput(fact_check_prediction_binary={})
            for _ in range(expected_count)
        ]

    # Single item methods for compatibility
    def forward(
        self,
        answer_triplets: list[list[str]],
        reference_triplets: list[list[list[str]]],
    ) -> FactCheckerOutput:
        """
        Process a single fact checking task (compatibility method).

        Args:
            answer_triplets: Single set of answer triplets
            reference_triplets: Single set of reference triplets

        Returns:
            FactCheckerOutput for the comparison
        """
        # Flatten reference triplets if needed
        if reference_triplets and isinstance(reference_triplets[0][0], list):
            reference_triplets = self.flatten_triplets(reference_triplets)

        batch_result = self.forward_batch_sync([answer_triplets], [reference_triplets])
        if batch_result.results:
            return batch_result.results[0]
        else:
            self.logger.warning(
                "Single item fact checking failed, returning empty result"
            )
            return FactCheckerOutput(fact_check_prediction_binary={})
