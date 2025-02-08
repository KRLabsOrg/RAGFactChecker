from rag_fact_checker.data import Config


class PipelineBase:
    """
    A base class for building data processing pipelines. Anything common to all pipelines should added here.

    Attributes:
        config (dict): A configuration dictionary containing parameters
                       for the pipeline's setup and behavior.
    """

    def __init__(self, config: Config):
        self.config = config
