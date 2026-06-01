"""Exo proxy backend for mlx-omni-server.

When MLX_OMNI_EXO_URL is set, mlx-omni-server proxies generation requests to
an Exo cluster (OpenAI-compatible API) instead of running mlx-lm locally.
Model loading is skipped entirely; the model is served by Exo across the cluster.

Usage:
    MLX_OMNI_EXO_URL=http://192.168.1.10:52415 mlx-omni-server --port 8082
"""

import json
import os
import time
import urllib.request
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from ...utils.logger import logger
from .core_types import (
    CompletionContent,
    CompletionResult,
    GenerationStats,
    StreamContent,
    StreamResult,
    ToolCall,
)
from .tools.qwen3_moe_tools_parser import Qwen3MoeToolParser

# Exo default endpoint
DEFAULT_EXO_URL = "http://localhost:52415"


class _ExoChatTemplate:
    """Minimal chat template shim so the Anthropic adapter can access tools_parser."""

    def __init__(self):
        self.tools_parser = Qwen3MoeToolParser()
        self.enable_thinking_parse = False

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("ExoChatTemplate does not apply templates directly")


class ExoChatGenerator:
    """Drop-in replacement for ChatGenerator that proxies to an Exo cluster.

    Implements the same generate() / generate_stream() interface as ChatGenerator
    so the Anthropic adapter works unchanged.
    """

    def __init__(self, exo_url: str, model_id: str):
        self._exo_url = exo_url.rstrip("/")
        self._model_id = model_id
        self.chat_template = _ExoChatTemplate()
        logger.info(f"ExoChatGenerator: proxying to {self._exo_url} model={model_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_openai_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        sampler: Optional[Dict[str, Any]],
        stream: bool,
        stop_words: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        # Exo rejects role=system inside messages; hoist to top-level "system" parameter.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        filtered_messages = [m for m in messages if m.get("role") != "system"]
        payload: Dict[str, Any] = {
            "model": self._model_id,
            "messages": filtered_messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if sampler:
            if "temp" in sampler:
                payload["temperature"] = sampler["temp"]
            if "top_p" in sampler:
                payload["top_p"] = sampler["top_p"]
            if "top_k" in sampler:
                payload["top_k"] = sampler["top_k"]
            if "min_p" in sampler:
                payload["min_p"] = sampler["min_p"]
        if stop_words:
            payload["stop"] = stop_words
        if tools:
            payload["tools"] = tools
        payload["use_prefix_cache"] = True
        # Pass through any extra kwargs Exo might support
        for k, v in kwargs.items():
            if v is not None and k not in payload:
                payload[k] = v
        return payload

    def _post(self, payload: Dict[str, Any], stream: bool):
        url = f"{self._exo_url}/v1/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=300)

    # ------------------------------------------------------------------
    # Public interface (matches ChatGenerator)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        sampler: Union[Dict[str, Any], Callable, None] = None,
        top_logprobs: Optional[int] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
        enable_prompt_cache: bool = False,
        **kwargs,
    ) -> CompletionResult:
        stop_words = kwargs.pop("_stop_words", [])
        # Also honour request stop_sequences passed via kwargs
        if isinstance(sampler, dict):
            sam = sampler
        else:
            sam = None

        payload = self._build_openai_request(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            sampler=sam,
            stream=False,
            stop_words=stop_words,
            **kwargs,
        )

        t0 = time.time()
        try:
            resp = self._post(payload, stream=False)
            body = json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(f"Exo request failed: {e}") from e

        choice = body["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")
        usage = body.get("usage", {})

        return CompletionResult(
            content=CompletionContent(text=text),
            finish_reason=finish_reason,
            stats=GenerationStats(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                generation_tps=0.0,
                time_to_first_token=time.time() - t0,
            ),
        )

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        sampler: Union[Dict[str, Any], Callable, None] = None,
        top_logprobs: Optional[int] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
        enable_prompt_cache: bool = False,
        **kwargs,
    ) -> Generator[StreamResult, None, None]:
        stop_words = kwargs.pop("_stop_words", [])
        if isinstance(sampler, dict):
            sam = sampler
        else:
            sam = None

        payload = self._build_openai_request(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            sampler=sam,
            stream=True,
            stop_words=stop_words,
            **kwargs,
        )

        t0 = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        chunk_index = 0
        accumulated_tool_calls: list = []

        try:
            resp = self._post(payload, stream=True)
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract usage if present (some servers send it in the last chunk)
                if "usage" in chunk and chunk["usage"]:
                    u = chunk["usage"]
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    completion_tokens = u.get("completion_tokens", completion_tokens)

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")
                text_delta = delta.get("content") or ""
                reasoning_delta = delta.get("reasoning_content") or ""

                # Accumulate OpenAI-format tool_calls from delta
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    while len(accumulated_tool_calls) <= idx:
                        accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        accumulated_tool_calls[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        accumulated_tool_calls[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        accumulated_tool_calls[idx]["arguments"] += fn["arguments"]

                if text_delta:
                    completion_tokens += 1
                    yield StreamResult(
                        content=StreamContent(
                            text_delta=text_delta,
                            chunk_index=chunk_index,
                            token=0,
                        ),
                        finish_reason=finish_reason,
                        stats=GenerationStats(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            time_to_first_token=time.time() - t0 if chunk_index == 0 else 0.0,
                        ),
                    )
                    chunk_index += 1
                elif finish_reason:
                    # Final chunk — emit dummy delta to carry stats
                    yield StreamResult(
                        content=StreamContent(
                            text_delta=" ",
                            chunk_index=chunk_index,
                            token=0,
                        ),
                        finish_reason=finish_reason,
                        stats=GenerationStats(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        ),
                    )

            # Yield accumulated OpenAI tool calls as XML text so the
            # Anthropic adapter's Qwen3MoeToolParser can process them
            valid_tcs = [tc for tc in accumulated_tool_calls if tc["name"]]
            for i, tc in enumerate(valid_tcs):
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except Exception:
                    args = {}
                xml_parts = [f"<tool_call>\n<function={tc['name']}>\n"]
                for k, v in args.items():
                    xml_parts.append(f"<parameter={k}>\n{v}\n</parameter>\n")
                xml_parts.append(f"</function>\n</tool_call>")
                xml = "".join(xml_parts)
                is_last = (i == len(valid_tcs) - 1)
                yield StreamResult(
                    content=StreamContent(
                        text_delta=xml,
                        chunk_index=chunk_index,
                        token=0,
                    ),
                    finish_reason="tool_calls" if is_last else None,
                    stats=GenerationStats(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ),
                )
                chunk_index += 1

        except Exception as e:
            raise RuntimeError(f"Exo stream failed: {e}") from e


def get_exo_generator(model_id: str) -> Optional[ExoChatGenerator]:
    """Return an ExoChatGenerator if MLX_OMNI_EXO_URL is set, else None."""
    exo_url = os.environ.get("MLX_OMNI_EXO_URL")
    if not exo_url:
        return None
    return ExoChatGenerator(exo_url=exo_url, model_id=model_id)
