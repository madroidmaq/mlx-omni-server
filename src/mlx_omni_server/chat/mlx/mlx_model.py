import time
import uuid
from typing import Any, Dict, Generator, Optional

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import GenerationResponse, stream_generate

from ...utils.logger import logger
from ..schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from ..text_models import BaseTextModel, GenerateResult
from .outlines_logits_processor import OutlinesLogitsProcessor
from .stop_tokens_checker import StopTokensChecker
from .tools.chat_tokenizer import ChatTokenizer


class MLXModel(BaseTextModel):
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

    def _stream_generate(
        self,
        prompt: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Generator[GenerationResponse, None, None]:
        try:
            tokenizer = self._chat_tokenizer.tokenizer
            stop_checker = None
            if request.stop:
                stop_checker = StopTokensChecker(
                    stop_words=request.stop,
                    tokenizer=tokenizer,
                )

            logits_processors = None
            if request.response_format and request.response_format.json_schema:
                logits_processors = [
                    OutlinesLogitsProcessor(
                        self._chat_tokenizer.tokenizer, request.response_format
                    )
                ]

            current_tokens = []
            last_text = ""

            max_completion_tokens = (
                request.max_completion_tokens
                or request.max_tokens
                or self._default_max_tokens
            )
            sampler = make_sampler(
                request.temperature or self._default_temperature,
                request.top_p or self._default_top_p,
            )

            for response in stream_generate(
                model=self._model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_completion_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                **kwargs,
            ):
                current_tokens.append(response.token)

                logprobs = None
                if request.logprobs:
                    logprobs = self._process_logprobs(
                        tokenizer, response, request.top_logprobs
                    )

                finish_reason = response.finish_reason or "length"
                should_trim = False
                if request.stop and stop_checker:
                    stop_condition = stop_checker.check_stop_condition(current_tokens)
                    if stop_condition.stop_met:
                        finish_reason = "stop"
                        if stop_condition.trim_length > 0:
                            current_tokens = current_tokens[
                                : -stop_condition.trim_length
                            ]
                            should_trim = True

                text = tokenizer.decode(current_tokens)
                delta_text = text[len(last_text) :]

                if delta_text or should_trim:
                    yield GenerateResult(
                        text=delta_text,
                        token=response.token,
                        finish_reason=finish_reason,
                        prompt_tokens=response.prompt_tokens,
                        generation_tokens=response.generation_tokens,
                        logprobs=logprobs,
                    )
                    last_text = text

                if should_trim:
                    break

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        try:
            completion = ""
            logprobs_result_list = []
            current_tokens = []
            finish_reason = "stop"
            result = None

            prompt = self._chat_tokenizer.encode(
                messages=request.messages,
                tools=request.tools,
                tool_choice=request.tool_choice if request.tool_choice else None,
            )
            logger.debug(f"Encoded prompt:\n{prompt}")

            params = self._get_generation_params(request)

            for result in self._stream_generate(
                prompt=prompt,
                request=request,
                **params,
            ):
                current_tokens.append(result.token)
                completion = self._chat_tokenizer.tokenizer.decode(current_tokens)

                if request.logprobs:
                    logprobs_result_list.append(result.logprobs)

                if not prompt:
                    prompt = result.text

                if result.finish_reason:
                    finish_reason = result.finish_reason

            if result is None:
                raise RuntimeError("No tokens generated")

            logger.debug(f"Model Response:\n{completion}")
            if request.tools:
                message = self._chat_tokenizer.decode(completion)
            else:
                message = ChatMessage(role=Role.ASSISTANT, content=completion)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=(
                            "tool_calls" if message.tool_calls else finish_reason
                        ),
                        logprobs=(
                            {"content": logprobs_result_list}
                            if logprobs_result_list
                            else None
                        ),
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.generation_tokens,
                    total_tokens=result.prompt_tokens + result.generation_tokens,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            params = self._get_generation_params(request)

            prompt = self._chat_tokenizer.encode(
                messages=request.messages,
                tools=request.tools,
            )
            logger.debug(f"Encoded prompt:\n{prompt}")

            completion = ""
            for result in self._stream_generate(
                prompt=prompt,
                request=request,
                **params,
            ):
                created = int(time.time())
                completion += result.text
                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT, content=result.text),
                            finish_reason=result.finish_reason,
                            logprobs=result.logprobs,
                        )
                    ],
                )

            if request.stream_options and request.stream_options.include_usage:
                created = int(time.time())
                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT),
                            finish_reason=None,
                            logprobs=None,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.generation_tokens,
                        total_tokens=result.prompt_tokens + result.generation_tokens,
                    ),
                )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise
