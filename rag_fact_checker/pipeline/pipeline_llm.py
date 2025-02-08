from langchain_openai import ChatOpenAI
from rag_fact_checker.data import Config
from rag_fact_checker.pipeline.pipeline_base import PipelineBase


class PipelineLLM(PipelineBase):
    """
    A pipeline class for interacting with Large Language Models (LLMs).
    Args:
        openai (openai.OpenAI): Module for direct interaction with OpenAI API
                                (not currently used).
        model (ChatOpenAI): An LLM model instance for generating outputs using
                            langchian_openai.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.model = ChatOpenAI(
            model=self.config.model.llm.generator_model,
            temperature=self.config.model.llm.temperature,
            max_retries=self.config.model.llm.request_max_try,
            api_key=self.config.model.llm.api_key,
        )
