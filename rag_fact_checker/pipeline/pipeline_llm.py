from openai import AsyncOpenAI, OpenAI

from rag_fact_checker.data import Config
from rag_fact_checker.pipeline.pipeline_base import PipelineBase


class PipelineLLM(PipelineBase):
    """
    A pipeline class for interacting with Large Language Models (LLMs).
    Args:
        model (OpenAI): A synchronous OpenAI client instance for generating outputs.
        async_model (AsyncOpenAI): An asynchronous OpenAI client instance for concurrent operations.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        if self.config.model.llm.base_url:
            self.model = OpenAI(
                api_key=self.config.model.llm.api_key,
                max_retries=self.config.model.llm.request_max_try,
                base_url=self.config.model.llm.base_url,
            )
            self.async_model = AsyncOpenAI(
                api_key=self.config.model.llm.api_key,
                max_retries=self.config.model.llm.request_max_try,
                base_url=self.config.model.llm.base_url,
            )
        else:
            self.model = OpenAI(
                api_key=self.config.model.llm.api_key,
                max_retries=self.config.model.llm.request_max_try,
            )
            self.async_model = AsyncOpenAI(
                api_key=self.config.model.llm.api_key,
                max_retries=self.config.model.llm.request_max_try,
            )
