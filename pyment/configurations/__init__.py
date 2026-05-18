"""Classes defining configurations for pyment training and inference."""

from .finetuning_configuration import FinetuningConfiguration
from .target_configuration import compile_target_encoder

__all__ = ['FinetuningConfiguration', 'compile_target_encoder']
