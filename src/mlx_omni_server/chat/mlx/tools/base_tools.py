import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from mlx_omni_server.utils.logger import logger

from ..core_types import ToolCall


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _extract_balanced_json_object(raw: str) -> Optional[str]:
    """Extract the first balanced JSON object substring from raw text.

    This is intentionally simple (stack-based) and handles arbitrarily nested braces.
    Returns None if no balanced object is found.
    """
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def _looks_like_nonempty_args_expected(tool_json_raw: str) -> bool:
    """Heuristic: if the raw tool JSON includes an arguments/parameters object that is not {}.

    Used to avoid returning ToolCall(..., arguments={}) when we actually failed to parse.
    """
    if '"arguments"' in tool_json_raw:
        # If explicitly empty, allow
        if re.search(r'"arguments"\s*:\s*\{\s*\}', tool_json_raw):
            return False
        return True
    if '"parameters"' in tool_json_raw:
        if re.search(r'"parameters"\s*:\s*\{\s*\}', tool_json_raw):
            return False
        return True
    return False


def _parse_tool_json_object(tool_json_raw: str) -> Optional[ToolCall]:
    try:
        payload = json.loads(tool_json_raw)
    except json.JSONDecodeError:
        balanced = _extract_balanced_json_object(tool_json_raw)
        if not balanced:
            return None
        try:
            payload = json.loads(balanced)
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    # HuggingFace unified tool use uses `arguments`; some variants use `parameters`.
    args = payload.get("arguments", payload.get("parameters", {}))
    if args is None:
        args = {}
    if not isinstance(args, dict):
        # Some models might emit arguments as a JSON-encoded string; try to decode.
        if isinstance(args, str):
            try:
                decoded = json.loads(args)
                if isinstance(decoded, dict):
                    args = decoded
                else:
                    args = {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = {}

    # Circuit breaker: if args were expected to be non-empty but ended up empty, treat as parse failure.
    if _looks_like_nonempty_args_expected(tool_json_raw) and not args:
        return None

    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)


def extract_tools(text: str) -> Optional[List[ToolCall]]:
    """Extract tool calls from text and return CoreToolCall objects directly.

    This parser is intentionally tolerant and supports arbitrarily nested JSON objects.

    Returns None if no tool calls are found or if tool-call parsing fails.
    """
    results: List[ToolCall] = []

    # Prefer explicit <tool_call>...</tool_call> blocks (HuggingFace unified tool-use).
    saw_tool_call_block = False
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        saw_tool_call_block = True
        raw = (m.group(1) or "").strip()
        tool_json = _extract_balanced_json_object(raw) or raw
        tool_call = _parse_tool_json_object(tool_json)
        if tool_call:
            results.append(tool_call)

    if results:
        return results

    # If the model attempted a <tool_call> but we couldn't parse it, do NOT fall back
    # to shallow regex parsing (it can produce empty args {} and trigger tool-error loops).
    if saw_tool_call_block:
        logger.warning("Detected <tool_call> block(s) but failed to parse; returning no tool calls")
        return None

    # Fallback: legacy regex-based extraction for non-tagged formats.
    # NOTE: this can fail on deeply nested braces; keep it as a best-effort path.
    pattern = (
        r'"name"\s*:\s*"([^"]+)"'  # Match name
        r"(?:"  # Start non-capturing group for optional arguments/parameters
        r"[^}]*?"  # Allow any characters in between
        r'(?:"arguments"|"parameters")'  # Match arguments or parameters
        r"\s*:\s*"  # Match colon and whitespace
        r"("  # Start capturing parameter value
        r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"  # Match nested objects (shallow)
        r"|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]"  # Match arrays
        r"|null"  # Match null
        r'|"[^"]*"'  # Match strings
        r")"  # End capturing
        r")?"  # Make the entire arguments/parameters section optional
    )

    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        name, args_str = match.groups()
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments as JSON: {args_str}")
            arguments = {}
        tool_call = ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)
        results.append(tool_call)

    return results if results else None


class BaseToolParser(ABC):
    start_tool_calls: str
    end_tool_calls: str

    @abstractmethod
    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        pass
