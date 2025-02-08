import logging
from typing import List, Dict

from langchain_core.messages import BaseMessage

from rag_fact_checker.data import FactCheckerOutput, Config
from rag_fact_checker.model.fact_checker import FactChecker
from rag_fact_checker.pipeline import PipelineLLM, PipelineDemonstration


class LLMMultiShotFactChecker(FactChecker, PipelineLLM, PipelineDemonstration):
    """
    LLMFactChecker is designed to compare answer triplets with reference triplets using a language model.
    The model compares a answer triplet with the reference triplets with LLM and returns the merged comparison result. Plus, it has the inquiry mode which adds reasons for final prediction in CoT sytled explanation.


    Attributes:
        config (Config): Config data class for initializing the class.
        logger (logging.Logger): Logger object for logging.

    Methods:

        forward(answer_triplets: List[List[str]], reference_triplets: List[List[List[str]]]): -> FactCheckerOutput
            Compares answer triplets with reference triplets and returns the comparison results.

        get_model_prompt(answer_triplets: List[List[str]], reference_triplets: List[List[str]], **kwargs) -> List[BaseMessage]:
            Generates a model prompt for comparing answer triplets with reference triplets.

        get_inquiry_model_prompt(answer_triplets: List[List[str]], reference_triplets: List[List[str]], **kwargs) -> List[BaseMessage]:
            Generates a model prompt for comparing answer triplets with reference triplets with inquiry mode.

        multishot_triplet_comparison_input_formatter(answer_triplets: List[List[str]], reference_triplets: List[List[str]]) -> Dict[str, str]:
            Formats the input for the multishot triplet comparison.

        parse_triplet_comparison_output(string_output: str) -> dict[int, bool]:
            Parses the output from the fact-checking model and returns the comparison result.

        parse_triplet_comparison_inquiry_output(string_output: str) -> dict[int, bool]:
            Parses the output from the inquiry mode fact-checking model and returns the comparison result.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        # Initialize all parent classes with the provided configuration
        FactChecker.__init__(self, config, logger)
        PipelineLLM.__init__(self, config)
        PipelineDemonstration.__init__(self, config)

    def forward(
        self,
        answer_triplets: List[List[str]],
        reference_triplets: List[List[List[str]]],
    ) -> FactCheckerOutput:
        """
        Perform a forward pass to fact-check the given answer triplets against reference triplets.

        Args:
            answer_triplets (list): The triplets generated by a model or user.
            reference_triplets (list): The ground-truth or reference triplets.


        Returns:
            FactCheckerOutput which contains the binary fact-checking results.
        """
        if self.config.model.fact_checker.split_reference_triplets:
            output_list = []
            for segment in reference_triplets:
                self.logger.debug(
                    "Segment: %s",
                    "\n-".join(
                        [
                            f"{idx} : {str(triplet)}"
                            for idx, triplet in enumerate(segment)
                        ]
                    ),
                )
                self.logger.debug("segment length: %s", len(segment))
                fact_check_prediction = self.model_forward(
                    answer_triplets=answer_triplets,
                    reference_triplets=segment,
                )
                output_list.append(fact_check_prediction)
            return self.merge_segment_outputs(output_list)
        else:
            reference_triplets = self.flatten_triplets(reference_triplets)
            return self.model_forward(
                answer_triplets,
                reference_triplets,
            )

    def model_forward(
        self,
        answer_triplets: List[List[str]],
        reference_triplets: List[List[str]],
    ) -> FactCheckerOutput:

        # Build the prompt for the model by formatting the input triplets
        if self.config.model.fact_checker.inquiry_mode:
            triplet_comparison_prompt = self.get_inquiry_model_prompt(
                answer_triplets,
                reference_triplets,
            )
        else:
            triplet_comparison_prompt = self.get_model_prompt(
                answer_triplets,
                reference_triplets,
            )
        # Invoke the LLM with the constructed prompt to get the raw matching result as text
        match_result = self.model.invoke(triplet_comparison_prompt).content
        # Parse the raw string output into a structured dictionary of triplet_idx: boolean_result

        if self.config.experiment_setup.log_prompts:
            self.logger.debug(triplet_comparison_prompt)

        if self.config.model.fact_checker.inquiry_mode:
            return FactCheckerOutput(
                fact_check_prediction_binary=self.parse_triplet_comparison_inquiry_output(
                    match_result
                )
            )
        else:
            return FactCheckerOutput(
                fact_check_prediction_binary=self.parse_triplet_comparison_output(
                    match_result
                )
            )

    def get_model_prompt(
        self,
        answer_triplets: List[List[str]],
        reference_triplets: List[List[str]],
        **kwargs,
    ) -> List[BaseMessage]:

        # Use the template message with the formatted input (answer and reference triplets)
        return self.message_list_template["n_shot_triplet_match_test"].invoke(
            input=self.multishot_triplet_comparison_input_formatter(
                answer_triplets,
                reference_triplets,
            )
        )

    def get_inquiry_model_prompt(
        self,
        answer_triplets: List[List[str]],
        reference_triplets: List[List[str]],
        **kwargs,
    ) -> List[BaseMessage]:
        # Use the template message with the formatted input (answer and reference triplets)
        return self.message_list_template["n_shot_triplet_match_test_inquiry"].invoke(
            input=self.multishot_triplet_comparison_input_formatter(
                answer_triplets,
                reference_triplets,
            )
        )

    def multishot_triplet_comparison_input_formatter(
        self,
        answer_triplets: List[List[str]],
        reference_triplets: List[List[str]],
    ) -> Dict[str, str]:
        """
        Format answer and reference triplets into strings suitable for LLM input.

        Args:
            fact_checker_input:FactCheckerInput

        Returns:
            MultishotFactCheckerModelInput
        """
        examples = self.get_demo_data(
            demo_type="fact_checker",
        )
        return {
            "answer_triplets": "\n-".join(
                [
                    f"{idx}: " + str(input_triplet)
                    for idx, input_triplet in enumerate(answer_triplets)
                ]
            ),
            "reference_triplets": "\n-".join(
                [
                    f"{idx}: " + str(source_triplet)
                    for idx, source_triplet in enumerate(reference_triplets)
                ]
            ),
            "examples": examples,
        }

    def parse_triplet_comparison_output(self, string_output: str) -> dict[int, bool]:
        """
        Parse the raw string output from the LLM into a structured dictionary of triplet results.

        The output should match the format: triplet_idx:result (e.g., "0:True, 1:False").

        Args:
            string_output (str): The raw output string from the LLM.

        Returns:
            dict: A dictionary where keys are triplet indices (int) and values are booleans indicating True/False for each triplet.
        """
        # Split the output by commas to separate each triplet's result
        splitted_string_outputs = string_output.replace("\n", ",").split(",")
        match_output = {}
        # Try to evaluate each part as a dictionary entry like "{0:True}"
        for splitted_string_output in splitted_string_outputs:
            try:
                # Remove potential hyphens and wrap in braces to form a valid Python dictionary entry
                match_output.update(
                    eval("{" + splitted_string_output.replace("-", "") + "}")
                )
            except Exception as e:
                # If parsing fails, skip that entry
                self.logger.warning(
                    f"Failed to parse fact checking output: '{string_output}'. Skipping it"
                )
                self.logger.debug("Error occured in : %s", string_output)
                pass
        return match_output

    def parse_triplet_comparison_inquiry_output(
        self, string_output: str
    ) -> dict[int, bool]:
        """
        Parse the raw string output from the LLM into a structured dictionary of triplet results.

        The output should match the format: triplet_idx:result (e.g., "0:True, 1:False").

        Args:
            string_output (str): The raw output string from the LLM.

        Returns:
            dict: A dictionary where keys are triplet indices (int) and values are booleans indicating True/False for each triplet.
        """
        # Split the output by commas to separate each triplet's result
        reference_triplets_part = string_output.split("[FINAL ANSWER]")[0]
        fact_check_result_part = string_output.split("[FINAL ANSWER]")[-1]
        splitted_string_outputs = (
            fact_check_result_part.replace("\n", ",")
            .replace("triplet_idx_", "")
            .replace("triplet_", "")
            .split(",")
        )
        match_output = {}
        # Try to evaluate each part as a dictionary entry like "{0:True}"
        if len(reference_triplets_part) > 0:
            self.logger.debug("Reference triplets: %s", reference_triplets_part)
        for splitted_string_output in splitted_string_outputs:
            try:
                # Remove potential hyphens and wrap in braces to form a valid Python dictionary entry
                match_output.update(
                    eval("{" + splitted_string_output.replace("-", "") + "}")
                )
            except Exception as e:
                # If parsing fails, skip that entry
                self.logger.warning(
                    f"Failed to parse fact checking output: '{fact_check_result_part}'. Skipping it"
                )
                self.logger.debug("Error occured in : %s", fact_check_result_part)
                pass
        return match_output
