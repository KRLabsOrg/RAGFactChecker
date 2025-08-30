import asyncio
import json
import logging

from rag_fact_checker.data import Config, TripletGeneratorOutput
from rag_fact_checker.model.triplet_generator import TripletGenerator
from rag_fact_checker.pipeline import PipelineLLM, PipelinePrompt
from rag_fact_checker.pipeline.batch_processor import BatchProcessingMixin, BatchResult


class BatchTripletGenerator(
    TripletGenerator, PipelineLLM, PipelinePrompt, BatchProcessingMixin
):
    """
    Batch-enabled triplet generator that can process multiple texts efficiently.

    Features:
    - Batch processing of multiple input texts
    - Async and sync batch processing
    - Configurable batch sizes and concurrency
    - Automatic retry and error handling
    - Structured JSON output for reliable parsing
    """

    def __init__(self, config: Config, logger: logging.Logger):
        TripletGenerator.__init__(self, config, logger)
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    def forward_batch_sync(
        self, input_texts: list[str]
    ) -> BatchResult[TripletGeneratorOutput]:
        """
        Process multiple input texts synchronously in batches.

        Args:
            input_texts: List of input texts to generate triplets from

        Returns:
            BatchResult containing TripletGeneratorOutput for each successful input
        """
        return self.process_items_in_batches_sync(
            input_texts, self._process_triplet_batch, "input_texts"
        )

    async def forward_batch_async(
        self, input_texts: list[str]
    ) -> BatchResult[TripletGeneratorOutput]:
        """
        Process multiple input texts asynchronously in batches.

        Args:
            input_texts: List of input texts to generate triplets from

        Returns:
            BatchResult containing TripletGeneratorOutput for each successful input
        """
        return await self.process_items_in_batches_async(
            input_texts, self._process_triplet_batch_async, "input_texts"
        )

    def _process_triplet_batch(
        self, input_texts: list[str]
    ) -> list[TripletGeneratorOutput]:
        """
        Process a batch of input texts synchronously.

        Args:
            input_texts: Batch of input texts

        Returns:
            List of TripletGeneratorOutput for each input
        """
        # Create batch prompt
        batch_prompt = self._create_batch_prompt(input_texts)

        # Define JSON schema for batch triplet output
        batch_triplet_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "batch_triplet_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "batch_results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "input_index": {"type": "integer"},
                                    "triplets": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 3,
                                            "maxItems": 3,
                                        },
                                    },
                                },
                                "required": ["input_index", "triplets"],
                                "additionalProperties": False,
                            },
                            "description": "Array of triplet results for each input text",
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
            response_format=batch_triplet_schema,
        )

        batch_output = response.choices[0].message.content
        return self._parse_batch_triplet_output(batch_output, len(input_texts))

    async def _process_triplet_batch_async(
        self, input_texts: list[str]
    ) -> list[TripletGeneratorOutput]:
        """
        Process a batch of input texts asynchronously.

        Args:
            input_texts: Batch of input texts

        Returns:
            List of TripletGeneratorOutput for each input
        """
        # For now, wrap sync call in async - could be enhanced with true async OpenAI client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._process_triplet_batch, input_texts
        )

    def _create_batch_prompt(self, input_texts: list[str]) -> list[dict[str, str]]:
        """
        Create a prompt for batch processing multiple input texts.

        Args:
            input_texts: List of input texts to process

        Returns:
            Formatted prompt for batch processing
        """
        # Format inputs with indices
        indexed_inputs = []
        for idx, text in enumerate(input_texts):
            indexed_inputs.append(f"Input {idx}: {text}")

        batch_input_text = "\n\n".join(indexed_inputs)

        template_names = self.message_list_template["batch_triplet_generation"]
        return self.create_messages(
            template_names,
            batch_input_text=batch_input_text,
            num_inputs=str(len(input_texts)),
        )

    def _parse_batch_triplet_output(
        self, batch_output: str, expected_count: int
    ) -> list[TripletGeneratorOutput]:
        """
        Parse JSON batch output into individual TripletGeneratorOutput objects.

        Args:
            batch_output: JSON string from the model
            expected_count: Expected number of results

        Returns:
            List of TripletGeneratorOutput objects
        """
        try:
            data = json.loads(batch_output.strip())
            batch_results = data.get("batch_results", [])

            # Initialize results array with default triplets
            results = [
                TripletGeneratorOutput(triplets=[self.default_triplet])
                for _ in range(expected_count)
            ]

            # Fill in actual results
            for result in batch_results:
                if (
                    isinstance(result, dict)
                    and "input_index" in result
                    and "triplets" in result
                ):
                    idx = result["input_index"]
                    if 0 <= idx < expected_count:
                        triplets = result["triplets"]

                        # Validate and clean triplets
                        validated_triplets = []
                        for triplet in triplets:
                            if isinstance(triplet, list) and len(triplet) == 3:
                                validated_triplets.append(
                                    [str(item) for item in triplet]
                                )
                            else:
                                self.logger.warning(
                                    f"Invalid triplet in batch result {idx}: {triplet}"
                                )
                                validated_triplets.append(self.default_triplet)

                        if validated_triplets:
                            results[idx] = TripletGeneratorOutput(
                                triplets=validated_triplets
                            )
                    else:
                        self.logger.warning(
                            f"Invalid input index in batch result: {idx}"
                        )
                else:
                    self.logger.warning(f"Invalid batch result structure: {result}")

            return results

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Error parsing batch triplet output: {str(e)}")
            self.logger.debug(f"Raw batch output: {batch_output}")
            # Return default triplets for all inputs
            return [
                TripletGeneratorOutput(triplets=[self.default_triplet])
                for _ in range(expected_count)
            ]
        except Exception as e:
            self.logger.warning(
                f"Unexpected error parsing batch triplet output: {str(e)}"
            )
            self.logger.debug(f"Raw batch output: {batch_output}")
            return [
                TripletGeneratorOutput(triplets=[self.default_triplet])
                for _ in range(expected_count)
            ]

    @property
    def default_triplet(self) -> list[str]:
        """Generate a default triplet."""
        return ["", "", ""]

    # Single item methods for compatibility
    def forward(self, input_text: str) -> TripletGeneratorOutput:
        """
        Process a single input text (compatibility method).

        Args:
            input_text: Single input text

        Returns:
            TripletGeneratorOutput for the input
        """
        batch_result = self.forward_batch_sync([input_text])
        if batch_result.results:
            return batch_result.results[0]
        else:
            self.logger.warning("Single item processing failed, returning default")
            return TripletGeneratorOutput(triplets=[self.default_triplet])
