"""
Prompt Cache Management Module

This module provides functionality for managing and optimizing model prompt caching,
to improve performance in multi-turn conversations.
"""

from dataclasses import dataclass, field
from typing import Any, List

from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_omni_server.chat.mlx.model_types import MLXModel

from ...utils.logger import logger


def common_prefix_len(list1, list2):
    """
    Calculates the length of the common prefix of two lists.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The length of the common prefix. Returns 0 if lists are empty
        or do not match at the first element.
    """
    # Determine the maximum possible length of the common prefix
    min_len = min(len(list1), len(list2))

    # Iterate up to the length of the shorter list
    for i in range(min_len):
        if list1[i] != list2[i]:
            # Mismatch found, the common prefix length is the current index
            return i

    # No mismatch found within the bounds of the shorter list,
    # so the common prefix length is the length of the shorter list.
    return min_len


@dataclass
class PromptCache:
    """
    Prompt cache class for storing and managing model prompt caches

    Attributes:
        tokens: Cached token sequence
        cache: Model's KV cache state, a list matching the number of model layers
        model_key: Model identifier to ensure cache matches the model
    """

    tokens: List[int] = field(default_factory=list)
    cache: List[Any] = field(default_factory=list)
    model_key: str = ""

    def extend_completion_cache(self, completion_tokens):
        self.tokens.extend(completion_tokens)

    def append_token(self, token):
        self.tokens.append(token)

    def reset_prompt_cache(self, model: MLXModel, prompt):
        logger.debug("*** Resetting cache. ***")
        self.model_key = model.model_id
        self.cache = make_prompt_cache(model.model)

        if model.draft_model is not None:
            self.cache += make_prompt_cache(model.draft_model)

        self.tokens = list(prompt)  # Cache the new prompt fully

    def get_prompt_cache(self, model, prompt):
        """
        Determines the portion of the prompt that needs processing by comparing
        it to the cached prompt and attempting to reuse the common prefix.

        This function updates the internal prompt cache state (tokens and model cache)
        based on the comparison. If a common prefix exists, it attempts to trim
        the model cache (if supported) to match the common prefix length, avoiding
        recomputation.

        Args:
            prompt (List[int]): The tokenized new prompt.

        Returns:
            Tuple[List[int], int]: A tuple where:
                - The first element is the suffix of the prompt that actually needs
                to be processed by the model
                - The second element is the length of the cached prefix that was
                successfully reused (0 if cache was reset)
        """
        cache_len = len(self.tokens)
        prompt_len = len(prompt)
        com_prefix_len = common_prefix_len(self.tokens, prompt)
        prompt_cached_tokens = 0

        # Leave at least one token in the prompt
        com_prefix_len = min(com_prefix_len, len(prompt))

        # Condition 1: Model changed or no common prefix at all. Reset cache.
        if self.model_key != model.model_id or com_prefix_len == 0:
            self.reset_prompt_cache(model, prompt)
        # Condition 2: Common prefix exists and matches cache length. Process suffix.
        elif com_prefix_len == cache_len:
            # When prompt exactly matches cache length, we're processing a complete reuse
            # of the cached prefix, so we only need to reprocess the very last token
            # The cached_prefix_length represents the full common prefix length
            # which is used for tracking purposes but we don't extend the cache here
            logger.debug(
                f"*** Cache is prefix of prompt (cache_len: {cache_len}, prompt_len: {prompt_len}). Processing suffix. ***"
            )
            if com_prefix_len == prompt_len:
                # Reuse same prompt - only need to process the last token
                # (triggers reprocessing of last exchange but caches all previous tokens)
                prompt = prompt[-1:]  # feed last token only
            else:
                # Normal case - process suffix that wasn't cached
                prompt = prompt[com_prefix_len:]
                self.tokens.extend(prompt)
            prompt_cached_tokens = com_prefix_len

        # Condition 3: Common prefix exists but is shorter than cache length. Attempt trim.
        elif com_prefix_len < cache_len:
            logger.debug(
                f"*** Common prefix ({com_prefix_len}) shorter than cache ({cache_len}). Attempting trim. ***"
            )

            # Back off 1 token: ensures last matching token is *not* in cache
            # Model re-sees transition (e.g., 'assistant' -> '\n') -> correct tool calls
            #
            # Example 1: Qwen with '\n' after 'assistant'
            #   Cache:  [... 'assistant' '\n' '<tool_call>']
            #   Prompt: [... 'assistant' '\n' '<tool_call>']
            #   com_prefix_len = 5 (includes '\n')
            #   safe_len = 4 -> cache ends at 'assistant'
            #   Prompt fed: '\n' '<tool_call>' -> model sees 'assistant' -> '\n' -> emits <tool_call>
            #
            # Example 2: No whitespace (hypothetical model)
            #   Cache:  [... '<|assistant|>']
            #   Prompt: [... '<|assistant|>' '<tool_call>']
            #   com_prefix_len = 4
            #   safe_len = 3 -> cache ends before role marker
            #   Prompt fed: '<|assistant|>' '<tool_call>' -> full transition
            #
            # Example 3: JSON mode
            #   Cache:  ['{' '"response":' '"']
            #   Prompt: ['{' '"response":' '"' '{' '"answer":' '42' '}']
            #   com_prefix_len = 3
            #   safe_len = 2 -> cache ends at '"response":'
            #   Prompt fed: '"' '{' ... -> model sees ':' -> '{'-> valid JSON

            safe_len = max(0, com_prefix_len - 1)
            num_trim = cache_len - safe_len

            # Trim the token cache
            self.tokens = self.tokens[:safe_len]

            # Attempt to trim the prompt cache if possible
            if num_trim > 0 and can_trim_prompt_cache(self.cache):
                trimmed_count = trim_prompt_cache(self.cache, num_trim)
                if not trimmed_count:
                    logger.debug("Trimming failed. Resetting cache")
                    self.reset_prompt_cache(model, prompt)
                    return prompt, prompt_cached_tokens

                # Determine the new prompt based on what's left in cache
                prompt_suffix = prompt[safe_len:]

                if prompt_suffix:
                    # Extend the tokens with the suffix that was not cached
                    prompt = prompt_suffix
                    self.tokens.extend(prompt_suffix)
                elif safe_len > 0:
                    # No suffix to work with - use the last cached token
                    prompt = self.tokens[-1:]

                prompt_cached_tokens = safe_len
            else:
                self.reset_prompt_cache(model, prompt)

        # This case should logically not be reached if com_prefix_len <= cache_len
        else:
            logger.error(
                f"Unexpected cache state: com_prefix_len ({com_prefix_len}) > cache_len ({cache_len}). Resetting cache."
            )
            self.reset_prompt_cache(model, prompt)

        logger.debug(f"Returning {len(prompt)} tokens for processing.")
        return prompt, prompt_cached_tokens
