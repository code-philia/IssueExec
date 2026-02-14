"""
Model abstraction layer for various LLM backends.

This module provides a unified interface for making completions requests to
different LLM providers including OpenAI, Anthropic, and DeepSeek.
"""

import json
from abc import ABC, abstractmethod
from typing import List

from .api_requests import (
    create_anthropic_config,
    create_chatgpt_config,
    request_anthropic_engine,
    request_chatgpt_engine,
)


class DecoderBase(ABC):
    """
    Abstract base class for LLM decoders.
    
    Provides a common interface for generating code completions from
    different language model backends.
    """
    
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        """
        Initialize the decoder.
        
        Args:
            name: Name of the model
            logger: Logger instance for logging
            batch_size: Number of completions to generate per request
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate
        """
        logger.info("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        """
        Generate code completions for the given message.
        
        Args:
            message: Input message/prompt
            num_samples: Number of samples to generate
            prompt_cache: Whether to use prompt caching
            
        Returns:
            List of dictionaries containing responses and usage information
        """
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        """
        Check if this decoder supports direct completion mode.
        
        Returns:
            True if direct completion is supported, False otherwise
        """
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    """
    Decoder for OpenAI chat models (GPT-4, GPT-3.5, etc.).
    """
    
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        """
        Generate completions using OpenAI's chat API.
        
        Args:
            message: Input message/prompt
            num_samples: Number of samples to generate
            prompt_cache: Whether to use prompt caching (not used for OpenAI)
            
        Returns:
            List of dictionaries containing responses and usage information
        """
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        # When generating multiple samples from the same input,
        # the input tokens are only charged once according to OpenAI API.
        # Therefore, we assume the request cost is only counted for the first sample.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicChatDecoder(OpenAIChatDecoder):
    """
    Decoder for Anthropic Claude models.
    
    Inherits from OpenAIChatDecoder since Claude uses an OpenAI-compatible
    proxy interface, making the calling method identical.
    """
    
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    """
    Decoder for DeepSeek models.
    """
    
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        """
        Generate completions using DeepSeek's API.
        
        Args:
            message: Input message/prompt
            num_samples: Number of samples to generate
            prompt_cache: Whether to use prompt caching (not used for DeepSeek)
            
        Returns:
            List of dictionaries containing responses and usage information
        """
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    """
    Factory function to create a decoder instance.
    
    Args:
        model: Name of the model to use
        backend: Backend type ("openai", "anthropic", or "deepseek")
        logger: Logger instance
        batch_size: Number of completions per request
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        A decoder instance for the specified backend
        
    Raises:
        NotImplementedError: If the backend is not supported
    """
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise NotImplementedError(f"Backend '{backend}' is not supported")
