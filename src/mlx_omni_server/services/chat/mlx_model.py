import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import GenerationResponse, stream_generate

from ...schemas.chat_schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from ...utils.logger import logger
from .base_models import BaseMLXModel, GenerateResult
from .tools.chat_tokenizer import ChatTokenizer


class MLXModel(BaseMLXModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(self, model_id: str, model, tokenizer: ChatTokenizer):
        self._model_id = model_id
        self._model = model
        self._default_max_tokens = 2048
        self._default_temperature = 1.0
        self._default_top_p = 1.0
        self._chat_tokenizer = tokenizer
        logger.info(f"Initialized MLXModel with model_id: {model_id}")

    def _get_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Extract and validate generation parameters from request"""
        return {}

    def _process_logprobs(
        self,
        tokenizer: TokenizerWrapper,
        response: GenerationResponse,
        top_k: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Process logprobs information from generation response to match OpenAI format"""
        current_token = response.token
        current_logprobs = response.logprobs

        # Get current token info
        token_str = tokenizer.decode([current_token])
        token_logprob = current_logprobs[current_token].item()
        token_bytes = token_str.encode("utf-8")

        # Base token info
        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_bytes),
        }

        # Process top logprobs
        top_logprobs = []
        if top_k is not None:
            # Get indices of top_k tokens
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = current_logprobs[top_indices]

            # Create detailed token information for each top token
            for idx, logprob in zip(top_indices.tolist(), top_probs.tolist()):
                token = tokenizer.decode([idx])
                token_bytes = token.encode("utf-8")
                top_logprobs.append(
                    {"token": token, "logprob": logprob, "bytes": list(token_bytes)}
                )

        return {**token_info, "top_logprobs": top_logprobs}

    async def _stream_generate(
        self,
        prompt: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> AsyncGenerator[GenerateResult, None]:
        try:
            tokenizer = self._chat_tokenizer.tokenizer
            for response in stream_generate(
                model=self._model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=request.max_completion_tokens
                or request.max_tokens
                or self._default_max_tokens,
                sampler=make_sampler(
                    request.temperature or self._default_temperature,
                    request.top_p or self._default_top_p,
                ),
                **kwargs,
            ):
                text = response.text

                # Process logprobs if available
                logprobs = None
                if request.logprobs:
                    logprobs = self._process_logprobs(
                        tokenizer, response, request.top_logprobs
                    )

                yield GenerateResult(
                    text=text,
                    token=response.token,
                    finished=False,
                    prompt_tokens=response.prompt_tokens,
                    generation_tokens=response.generation_tokens,
                    logprobs=logprobs,
                )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise

    def _build_chat_response(
        self,
        model: str,
        message: ChatMessage,
        usage: ChatCompletionUsage,
        logprobs: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason="tool_calls" if message.tool_calls else "stop",
                    logprobs=logprobs,
                )
            ],
            usage=usage,
        )

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate a chat completion response."""
        try:
            completion = ""
            logprobs_result_list = []

            # Format prompt with tools if provided
            prompt = self._chat_tokenizer.encode(
                messages=request.messages,
                tools=request.tools,
                tool_choice=request.tool_choice if request.tool_choice else None,
            )
            logger.debug(f"Encoded prompt:\n{prompt}")

            params = self._get_generation_params(request)

            async for result in self._stream_generate(
                prompt=prompt,
                request=request,
                **params,
            ):
                completion += result.text

                if request.logprobs:
                    logprobs_result_list.append(result.logprobs)

                if not prompt:  # Get prompt token count on first iteration
                    prompt = result.text

            logger.debug(f"Model Response:\n{completion}")

            message = self._chat_tokenizer.decode(completion)

            return self._build_chat_response(
                model=request.model,
                message=message,
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.generation_tokens,
                    total_tokens=result.prompt_tokens + result.generation_tokens,
                ),
                logprobs=(
                    {"content": logprobs_result_list} if logprobs_result_list else None
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream generate chat completion chunks."""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            created = int(time.time())

            params = self._get_generation_params(request)

            # Format prompt with tools if provided
            prompt = self._chat_tokenizer.encode(
                messages=request.messages,
                tools=request.tools,
            )
            logger.debug(f"Encoded prompt:\n{prompt}")
            completion = ""
            async for result in self._stream_generate(
                prompt=prompt,
                request=request,
                **params,
            ):
                completion += result.text

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT, content=result.text),
                            finish_reason="stop" if result.finished else None,
                            logprobs=result.logprobs,
                        )
                    ],
                )
            logger.debug(f"Stream Model Response:\n{completion}")

        except Exception as e:
            logger.error(f"Failed to stream generate: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to stream generate: {str(e)}")
