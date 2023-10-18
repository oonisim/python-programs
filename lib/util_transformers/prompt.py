"""Huggingface dataset prompt module based on PromptSource
[Huggingface Datasets]
https://huggingface.co/docs/datasets/index
https://huggingface.co/docs/datasets/package_reference/logging_methods

[PromptSource]
https://github.com/bigscience-workshop/promptsource
"""
import json
import logging
from typing import (
    List,
    Dict,
    Tuple,
    Callable,
    Optional,
    Union,
    Iterable,
)

import datasets.utils.logging
from datasets import (
    load_dataset,
    get_dataset_split_names
)
from datasets.dataset_dict import (
    IterableDatasetDict
)
from promptsource.templates import (
    DatasetTemplates,
    Template
)

from util_logging import (
    get_logger,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)
datasets.utils.logging.set_verbosity(_logger.getEffectiveLevel())


# --------------------------------------------------------------------------------
# Prompt template class
# --------------------------------------------------------------------------------
class PromptTemplate:
    """Class for prompt template processing based on the Huggingface datasets and PromptSource.
    Currently only Streaming=True is used for Dataset.
    """
    # --------------------------------------------------------------------------------
    # Class
    # --------------------------------------------------------------------------------
    @staticmethod
    def exists_dataset(dataset_name: str) -> bool:
        """Check if the Huggingface has the dataset
        Args:
            dataset_name: name of the dataset
        Returns: True if the dataset exists
        """
        try:
            PromptTemplate.get_dataset_split_names(dataset_name=dataset_name)
            return True
        except FileNotFoundError as error:
            return False

    @staticmethod
    def get_dataset_split_names(dataset_name: str) -> List[str]:
        """Get the available splits in the Dataset
        Args:
            dataset_name: name of the dataset
        Returns: List of split names
        Raises: FileNotFoundError if thd dataset does not exist
        """
        return get_dataset_split_names(path=dataset_name)

    @staticmethod
    def exist_prompt_templates(dataset_name: str) -> bool:
        """Check if PromptSource has templates for the dataset
        Args:
            dataset_name: name of the dataset
        Returns: True if the templates exist for the dataset
        """
        try:
            PromptTemplate.get_prompt_templates(dataset_name=dataset_name)
            return True
        except RuntimeError:
            return False

    @staticmethod
    def get_prompt_templates(dataset_name: str) -> DatasetTemplates:
        """Get Prompt Templates
        Args:
            dataset_name: name of the dataset
        Returns: DatasetTemplates instance
        Raises: RuntimeError if there is no prompt templates
        """
        prompt_templates: DatasetTemplates = DatasetTemplates(dataset_name=dataset_name)
        if not prompt_templates.all_template_names:
            raise RuntimeError(f"prompt is not available for {dataset_name}.")

        return prompt_templates

    @staticmethod
    def get_prompt_template_names(dataset_name: str) -> List[str]:
        """Get names of the prompt templates
        Args:
            dataset_name: name of the dataset
        Returns: List of prompt template names
        """
        return PromptTemplate.get_prompt_templates(dataset_name=dataset_name).all_template_names

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------
    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        """Set up the Huggingface dataset and the corresponding prompt template.
        Args:
            dataset_name: name of the dataset
        Raises: RuntimeError if there is no dataset or no templates
        """
        if not self.exists_dataset(dataset_name=dataset_name):
            raise RuntimeError(f"{dataset_name} does not exist.")

        if not self.exist_prompt_templates(dataset_name=dataset_name):
            raise RuntimeError(f"no prompt templates for {dataset_name}.")

        # --------------------------------------------------------------------------------
        # Huggingface Dataset. Use streaming.
        # --------------------------------------------------------------------------------
        self._dataset_name: str = dataset_name
        self._dataset_split_names: List[str] = self.get_dataset_split_names(dataset_name=dataset_name)
        self._dataset: IterableDatasetDict = load_dataset(
            path=dataset_name,
            streaming=True
        )

        # --------------------------------------------------------------------------------
        # Prompt Source templates instance
        # --------------------------------------------------------------------------------
        self._prompt_templates: DatasetTemplates = self.get_prompt_templates(dataset_name=dataset_name)
        # Use the first template as the default prompt template
        self._prompt_template_name = self._prompt_templates.all_template_names[0]
        self._prompt_template: Template = self._prompt_templates[self._prompt_template_name]

        _logger.info(
            "configured the dataset [%s] with prompt templates %s.",
            self.dataset_name, self.prompt_template_names
        )

    @property
    def prompt_templates(self) -> DatasetTemplates:
        return self._prompt_templates

    @property
    def prompt_template_names(self) -> List[str]:
        assert len(self._prompt_templates.all_template_names) > 0
        return self.prompt_templates.all_template_names

    @property
    def prompt_template(self) -> Template:
        """Get the current prompt template instance"""
        return self._prompt_template

    @property
    def prompt_template_name(self) -> str:
        """Get current prompt template name
        """
        return self.prompt_template.get_name()

    def set_prompt_template(self, template_name: str):
        """Change the current prompt template"""
        assert template_name in self.prompt_template_names, \
            f"template {template_name} must be one of {self.prompt_template_names}."
        self._prompt_template = self.prompt_templates[template_name]

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            dataset_name: str
    ):
        self.dataset_name = dataset_name

    # --------------------------------------------------------------------------------
    # Prompt
    # --------------------------------------------------------------------------------
    def __call__(self, example: Dict, *args, **kwargs) -> List[str]:
        """Apply prompt template on the example
        Args:
            example: input to apply the prompt template
        Returns: prompt generated via the template
        """
        return self.prompt_template.apply(example=example, *args, **kwargs)

    def apply(self, example: Dict) -> str:
        """Apply prompt template on the example
        Args:
            example: input to apply the prompt template
        Returns: single prompt string generated via the template
        """
        return self.prompt_template.apply(example=example)[0]

