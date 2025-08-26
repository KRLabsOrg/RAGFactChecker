from openai import OpenAI

from rag_fact_checker.data import Config
from rag_fact_checker.pipeline.pipeline_base import PipelineBase


class PipelineLLM(PipelineBase):
    """
    A pipeline class for interacting with Large Language Models (LLMs).
    Args:
        model (OpenAI): An OpenAI client instance for generating outputs.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        if self.config.model.llm.base_url:
            self.model = OpenAI(
                api_key=self.config.model.llm.api_key,
                max_retries=self.config.model.llm.request_max_try,
                base_url=self.config.model.llm.base_url,
            )
        else:
            self.model = OpenAI(
                api_key=self.config.model.llm.api_key,
                max_retries=self.config.model.llm.request_max_try,
            )
