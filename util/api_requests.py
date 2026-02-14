# util/api_requests.py
"""
API Request Utilities

This module provides utilities for making API requests to various LLM backends,
including OpenAI, Anthropic, and DeepSeek. It supports API key rotation for
rate limit handling and connection error recovery.
"""

import os
import random
import threading
import time
from typing import Dict, Union

import anthropic
import openai
import tiktoken
import transformers


# ============== API Rotator ==============

class APIRotator:
    """
    API Key and URL Rotator (thread-safe, process-independent)
    
    Supports loading multiple API keys and URLs from environment variables
    (comma-separated), rotating to the next configuration when encountering
    connection errors or rate limits.
    
    Environment variable configuration examples:
        # Single configuration (backward compatible)
        OPENAI_API_KEY="sk-xxx"
        OPENAI_BASE_URL="https://api.example.com/v1"
        
        # Multiple configurations (comma-separated)
        OPENAI_API_KEY="sk-key1,sk-key2,sk-key3"
        OPENAI_BASE_URL="https://api1.example.com/v1,https://api2.example.com/v1"
        ANTHROPIC_API_KEY="sk-ant-key1,sk-ant-key2"
        ANTHROPIC_BASE_URL="https://api.example.com/v1"
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        # Singleton pattern (process-level singleton)
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Load configurations
        self.openai_keys = self._load_list_from_env("OPENAI_API_KEY")
        self.openai_urls = self._load_list_from_env("OPENAI_BASE_URL")
        self.anthropic_keys = self._load_list_from_env("ANTHROPIC_API_KEY")
        self.anthropic_urls = self._load_list_from_env("ANTHROPIC_BASE_URL")
        
        # Random starting index (distribute requests across multiple processes)
        self.openai_key_idx = random.randint(0, max(1, len(self.openai_keys))) - 1
        self.openai_url_idx = random.randint(0, max(1, len(self.openai_urls))) - 1
        self.anthropic_key_idx = random.randint(0, max(1, len(self.anthropic_keys))) - 1
        self.anthropic_url_idx = random.randint(0, max(1, len(self.anthropic_urls))) - 1
        
        self._rotate_lock = threading.Lock()
        self._initialized = True
    
    def _load_list_from_env(self, env_name: str) -> list:
        """Load comma-separated list from environment variable"""
        value = os.environ.get(env_name, "")
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]
    
    def get_openai_config(self) -> tuple:
        """Get current OpenAI API configuration (api_key, base_url)"""
        api_key = self.openai_keys[self.openai_key_idx] if self.openai_keys else None
        base_url = self.openai_urls[self.openai_url_idx] if self.openai_urls else None
        return api_key, base_url
    
    def rotate_openai(self) -> tuple:
        """Rotate to next OpenAI configuration, return new configuration"""
        with self._rotate_lock:
            old_key_idx, old_url_idx = self.openai_key_idx, self.openai_url_idx
            
            if len(self.openai_keys) > 1:
                self.openai_key_idx = (self.openai_key_idx + 1) % len(self.openai_keys)
            if len(self.openai_urls) > 1:
                self.openai_url_idx = (self.openai_url_idx + 1) % len(self.openai_urls)
        
        return self.get_openai_config()
    
    def get_anthropic_config(self) -> tuple:
        """Get current Anthropic API configuration (api_key, base_url)"""
        api_key = self.anthropic_keys[self.anthropic_key_idx] if self.anthropic_keys else None
        base_url = self.anthropic_urls[self.anthropic_url_idx] if self.anthropic_urls else None
        return api_key, base_url
    
    def rotate_anthropic(self) -> tuple:
        """Rotate to next Anthropic configuration, return new configuration"""
        with self._rotate_lock:
            old_key_idx, old_url_idx = self.anthropic_key_idx, self.anthropic_url_idx
            
            if len(self.anthropic_keys) > 1:
                self.anthropic_key_idx = (self.anthropic_key_idx + 1) % len(self.anthropic_keys)
            if len(self.anthropic_urls) > 1:
                self.anthropic_url_idx = (self.anthropic_url_idx + 1) % len(self.anthropic_urls)
        
        return self.get_anthropic_config()
    
    def has_rotation_available(self, api_type: str = "openai") -> bool:
        """Check if multiple configurations are available for rotation"""
        if api_type == "openai":
            return len(self.openai_keys) > 1 or len(self.openai_urls) > 1
        elif api_type == "anthropic":
            return len(self.anthropic_keys) > 1 or len(self.anthropic_urls) > 1
        return False


def get_rotator() -> APIRotator:
    """Get global API rotator instance"""
    return APIRotator()


# ============== Token Counting ==============

def num_tokens_from_messages(message, model: str = "gpt-3.5-turbo-0301") -> int:
    """
    Returns the number of tokens used by a list of messages.
    
    Args:
        message: The message content (string or list of message dicts)
        model: The model name for tokenizer selection
        
    Returns:
        Estimated number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if isinstance(message, list):
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    
    # Model-specific safety factor
    model_lower = model.lower()
    if "deepseek" in model_lower:
        num_tokens = int(num_tokens * 1.3)  # 30% safety margin
    elif "claude" in model_lower:
        # Claude uses different tokenizer, tiktoken underestimates by ~40-60%
        # Use 1.6x as conservative safety factor
        num_tokens = int(num_tokens * 1.6)
    
    return num_tokens


# ============== Configuration Creation ==============

def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    """
    Create configuration for ChatGPT API request.
    
    Args:
        message: The user message (string or list of message dicts)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Number of completions to generate
        system_message: System message to prepend
        model: Model name
        
    Returns:
        Configuration dictionary for API request
    """
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    """
    Create configuration for Anthropic API request.
    
    Args:
        message: The user message (string or list of message dicts)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Number of completions to generate (unused for Anthropic)
        system_message: System message (unused in this implementation)
        model: Model name
        tools: Optional list of tools for function calling
        
    Returns:
        Configuration dictionary for API request
    """
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


# ============== API Request Functions ==============

def request_chatgpt_engine(
    config: Dict,
    logger,
    base_url: str = None,
    max_retries: int = 40,
    timeout: int = 100
):
    """
    Make a request to the ChatGPT API with retry logic.
    
    Args:
        config: Configuration dictionary for the request
        logger: Logger instance
        base_url: Optional base URL override
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        API response or None if all retries failed
    """
    ret = None
    retries = 0
    
    # Determine if rotation is enabled: only when base_url is not explicitly specified
    use_rotation = (base_url is None)
    rotator = get_rotator() if use_rotation else None

    while ret is None and retries < max_retries:
        # Determine current key and url
        if use_rotation:
            current_key, current_url = rotator.get_openai_config()
        else:
            current_key = None  # Use environment variable default
            current_url = base_url
        
        # Create new client for each request (supports rotation with new config)
        client = openai.OpenAI(api_key=current_key, base_url=current_url)
        
        try:
            logger.info("Creating API request")
            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                logger.info("Rate limit exceeded. Waiting...")
                logger.info(e)
                if use_rotation:
                    rotator.rotate_openai()
                    logger.info("Rotated to next API key/URL")
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                logger.info("API connection error. Waiting...")
                logger.info(e)
                if use_rotation:
                    rotator.rotate_openai()
                    logger.info("Rotated to next API key/URL")
                time.sleep(5)
            else:
                logger.info("Unknown error. Waiting...")
                logger.info(e)
                if use_rotation:
                    rotator.rotate_openai()
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def request_anthropic_engine(
    config: Dict,
    logger,
    max_retries: int = 40,
    timeout: int = 500,
    prompt_cache: bool = False
):
    """
    Make a request to the Anthropic API with retry logic.
    
    Args:
        config: Configuration dictionary for the request
        logger: Logger instance
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        prompt_cache: Whether to use prompt caching
        
    Returns:
        API response or None if all retries failed
    """
    ret = None
    retries = 0
    
    rotator = get_rotator()

    while ret is None and retries < max_retries:
        current_key, current_url = rotator.get_anthropic_config()
        
        client_kwargs = {}
        if current_key:
            client_kwargs['api_key'] = current_key
        if current_url:
            client_kwargs['base_url'] = current_url
        
        client = anthropic.Anthropic(**client_kwargs)
        
        try:
            start_time = time.time()
            if prompt_cache:
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except anthropic.RateLimitError as e:
            logger.error("Rate limit exceeded. Waiting...", exc_info=True)
            logger.info(e)
            rotator.rotate_anthropic()
            logger.info("Rotated to next API key/URL")
            time.sleep(10 * (retries + 1))
        except anthropic.APIConnectionError as e:
            logger.error("API connection error. Waiting...", exc_info=True)
            logger.info(e)
            rotator.rotate_anthropic()
            logger.info("Rotated to next API key/URL")
            time.sleep(10 * (retries + 1))
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            logger.info(f"Unknown error: {e}")
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            rotator.rotate_anthropic()
            time.sleep(10 * (retries + 1))
        
        retries += 1

    return ret
