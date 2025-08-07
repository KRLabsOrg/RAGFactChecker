import logging

from rag_fact_checker.data import Config, TripletGeneratorOutput
from rag_fact_checker.model.triplet_generator.triplet_generator import (
    TripletGenerator,
)
from rag_fact_checker.pipeline import PipelineLLM, PipelinePrompt
from rag_fact_checker.pipeline.simple_batch_processor import (
    SimpleBatchProcessingMixin,
    SimpleBatchResult,
)


class LLMTripletGenerator(
    TripletGenerator, PipelineLLM, PipelinePrompt, SimpleBatchProcessingMixin
):
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

    get_model_prompt(self, text_input:str, **kwargs) -> List[Dict[str, str]]
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

        # Define JSON schema for structured triplet output
        triplet_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "triplet_generation_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "triplets": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "description": "Array of triplets, each containing [subject, predicate, object]",
                        }
                    },
                    "required": ["triplets"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = self.model.chat.completions.create(
            model=self.config.model.llm.generator_model,
            messages=triplet_generation_prompt,
            temperature=self.config.model.llm.temperature,
            response_format=triplet_schema,
        )
        triplet_generation_output = response.choices[0].message.content

        if self.config.experiment_setup.log_prompts:
            self.logger.debug(triplet_generation_prompt)

        return TripletGeneratorOutput(
            triplets=self.parse_triplet_generation_output(triplet_generation_output)
        )

    @property
    def default_triplet(self) -> list[str]:
        """
        Generates a default triplet.
        Returns:
            list: A list containing three empty strings.
        """
        return ["", "", ""]

    def get_model_prompt(self, input_text: str, **kwargs) -> list[dict[str, str]]:
        """
        Create a prompt for triplet generation using the provided text input.
        Args:
            input_text (Optional[str]): The input text for generating the prompt.
            **kwargs: Additional keyword arguments. Must include 'generated_answer' if text_input is not provided.

        Returns:
            List[Dict[str, str]]: The generated prompt for triplet generation.

        Raises:
            AssertionError: If neither input_text nor 'generated_answer' in kwargs is provided.
        """
        if input_text == None:
            assert "generated_answer" in kwargs, (
                "one of input_text input or generated_answer should be provided"
            )
            input_text = kwargs["generated_answer"]

        template_names = self.message_list_template["triplet_generation"]
        return self.create_messages(
            template_names, **self.triplet_generation_input_formatter(input_text)
        )

    def triplet_generation_input_formatter(self, input_text: str) -> dict[str, str]:
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
    ) -> list[list[str]]:
        """
        Parse JSON output to triplets.
        Args:
            triplet_generation_model_output (str): The JSON output containing triplets.

        Returns:
            list: A list of triplets parsed from the JSON output. If parsing fails, returns default triplet.
        """
        import json

        try:
            # Parse JSON output
            data = json.loads(triplet_generation_model_output.strip())
            triplets = data.get("triplets", [])

            # Validate triplets and fix any invalid ones
            result = []
            for triplet in triplets:
                if isinstance(triplet, list) and len(triplet) == 3:
                    result.append(
                        [str(item) for item in triplet]
                    )  # Ensure all elements are strings
                else:
                    self.logger.warning("Invalid triplet structure: %s", str(triplet))
                    result.append(self.default_triplet)

            # Return at least one triplet
            return result if result else [self.default_triplet]

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning("Error parsing JSON triplet output: %s", str(e))
            self.logger.debug("Raw triplet output: %s", triplet_generation_model_output)
            return [self.default_triplet]
        except Exception as e:
            self.logger.warning("Unexpected error parsing triplet output: %s", str(e))
            self.logger.debug("Raw triplet output: %s", triplet_generation_model_output)
            return [self.default_triplet]

    # Batch processing methods
    def forward_batch(
        self, input_texts: list[str]
    ) -> SimpleBatchResult[TripletGeneratorOutput]:
        """
        Process multiple input texts concurrently.

        Args:
            input_texts: List of input texts to generate triplets from

        Returns:
            SimpleBatchResult containing TripletGeneratorOutput for each successful input
        """
        return self.process_items_concurrently(input_texts, self.forward, "input_texts")

    async def forward_batch_async(
        self, input_texts: list[str]
    ) -> SimpleBatchResult[TripletGeneratorOutput]:
        """
        Process multiple input texts concurrently with async support.

        Args:
            input_texts: List of input texts to generate triplets from

        Returns:
            SimpleBatchResult containing TripletGeneratorOutput for each successful input
        """

        async def async_forward(text: str) -> TripletGeneratorOutput:
            import asyncio

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.forward, text)

        return await self.process_items_concurrently_async(
            input_texts, async_forward, "input_texts"
        )
