import json
from abc import abstractmethod

from easydict import EasyDict as edict

from rag_fact_checker.data import Config
from rag_fact_checker.pipeline.pipeline_base import PipelineBase


class PipelinePrompt(PipelineBase):
    """
    A pipeline for any class using prompts.
    You should inherit from this class if you are using prompts in your class.
    """

    def __init__(self, config: Config):
        """
        Initializes the PipelinePrompt class. Here we does the following:
        - Load prompt file
        - Define prompt templates based on the message type and template.
        - Get message list templates

        Args:
            config (edict): Configuration file
        """
        super().__init__(config)

        # Load prompts from JSON file instead of hardcoded constant
        with open(config.path.prompts) as f:
            prompt_bank = json.load(f)
        self.prompts = edict(prompt_bank)

        self.prompt_templates = self.get_prompt_templates()
        self.message_list_template = self.get_message_list_templates()

    def define_prompt_template(
        self, template_dict: dict, message_type: str
    ) -> dict[str, str]:
        """
        Defines a prompt template based on the message type.

        Args:
            template_dict (dict): Dictionary containing template details.
            message_type (str): Type of the message, e.g., 'human' or 'system'.

        Returns:
            Dict[str, str]: A dictionary with 'role' and 'format' keys.

        Raises:
            NotImplementedError: If the message type is not supported.
        """
        if message_type == "human":
            return {"role": "user", "format": template_dict["format"]}
        elif message_type == "system":
            return {"role": "system", "format": template_dict["format"]}
        else:
            raise NotImplementedError

    def get_prompt_templates(
        self,
    ) -> dict[str, dict[str, str]]:
        """
        Retrieves and constructs all prompt templates from the configuration.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary where keys are template names and values are prompt templates.
        """
        prompt_templates = {}
        for message_type, template_dicts in self.prompts.items():
            for template_name, template_dict in template_dicts.items():
                prompt_templates[template_name] = self.define_prompt_template(
                    template_dict, message_type
                )
        return prompt_templates

    def get_message_list_templates(self) -> dict[str, list[str]]:
        """
        Generates a dictionary of message list templates for different purposes.

        Returns:
            dict: A dictionary containing message list templates
        """
        message_list_template = {}
        for template_name, _ in self.prompts["human"].items():
            message_list_template[template_name] = [
                f"{template_name}_instruction",
                template_name,
            ]
        return message_list_template

    def format_message(self, template_name: str, **kwargs) -> str:
        """
        Formats a message template with the provided kwargs.

        Args:
            template_name (str): Name of the template to format
            **kwargs: Arguments to format the template with

        Returns:
            str: Formatted message
        """
        template = self.prompt_templates[template_name]
        return template["format"].format(**kwargs)

    def create_messages(
        self, template_names: list[str], **kwargs
    ) -> list[dict[str, str]]:
        """
        Creates a list of messages for OpenAI API from template names.

        Args:
            template_names (List[str]): List of template names to use
            **kwargs: Arguments to format templates with

        Returns:
            List[Dict[str, str]]: List of messages with 'role' and 'content' keys
        """
        messages = []
        for template_name in template_names:
            template = self.prompt_templates[template_name]
            content = template["format"].format(**kwargs)
            messages.append({"role": template["role"], "content": content})
        return messages

    @abstractmethod
    def get_model_prompt(self, **kwargs) -> list[dict[str, str]]:
        """
        Abstract method to be implemented in subclasses, defining how the model prompt is constructed.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Dict[str, str]]: List of messages for OpenAI API
        """
        raise NotImplementedError
