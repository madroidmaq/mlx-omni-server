"""Targeted tests for mlx_vlm model detection and wrappers."""

import sys
import types

from mlx_omni_server.chat.mlx import model_types
from mlx_omni_server.chat.mlx.model_types import MLXModel, _is_vlm_model, load_mlx_model


def test_is_vlm_model_detects_gemma4_config_and_model_id():
    assert _is_vlm_model("anything", {"model_type": "gemma4"}) is True
    assert _is_vlm_model("mlx-community/Gemma4-VLM", {"model_type": "unknown"}) is True
    assert _is_vlm_model("mlx-community/qwen3", {"model_type": "qwen3"}) is False


def test_load_mlx_model_uses_mlx_vlm_for_gemma4(monkeypatch):
    class RawVLMModel:
        config = {"model_type": "gemma4"}

        def __call__(self, *args, **kwargs):
            return "called"

    class RawTokenizer:
        def __len__(self):
            return 42

        def encode(self, text):
            return [1, 2, 3]

        def decode(self, tokens):
            return "decoded"

    class RawProcessor:
        tokenizer = RawTokenizer()

    captured_template_args = {}

    def fake_apply_chat_template(processor, config, messages, **kwargs):
        captured_template_args["processor"] = processor
        captured_template_args["config"] = config
        captured_template_args["messages"] = messages
        captured_template_args["kwargs"] = kwargs
        return "vlm prompt"

    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.load = lambda model_id: (RawVLMModel(), RawProcessor())
    fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    fake_prompt_utils.apply_chat_template = fake_apply_chat_template
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

    monkeypatch.setattr(model_types, "get_model_path", lambda model_id: "/fake/model")
    monkeypatch.setattr(
        model_types, "load_config", lambda model_path: {"model_type": "gemma4"}
    )

    class FakeChatTemplate:
        def __init__(self, model_type, tokenizer):
            self.model_type = model_type
            self.tokenizer = tokenizer

    fake_tools_module = types.ModuleType("mlx_omni_server.chat.mlx.tools.chat_template")
    fake_tools_module.ChatTemplate = FakeChatTemplate
    monkeypatch.setitem(
        sys.modules,
        "mlx_omni_server.chat.mlx.tools.chat_template",
        fake_tools_module,
    )

    loaded = load_mlx_model("test-gemma4-vlm", draft_model_id="draft")

    assert isinstance(loaded, MLXModel)
    assert loaded.is_vlm_model is True
    assert loaded.draft_model is None
    assert loaded.has_draft_model() is False
    assert loaded.model() == "called"
    assert loaded.tokenizer.vocab_size == 42
    assert loaded.tokenizer.encode("hi") == [1, 2, 3]
    assert loaded.tokenizer.decode([1]) == "decoded"

    prompt = loaded.tokenizer.apply_chat_template(
        messages=[{"role": "user", "content": "look"}], add_generation_prompt=True
    )
    assert prompt == "vlm prompt"
    assert captured_template_args["processor"] is loaded.model.processor
    assert captured_template_args["messages"] == [{"role": "user", "content": "look"}]
    assert captured_template_args["kwargs"] == {
        "add_generation_prompt": True,
        "num_images": 0,
        "num_audios": 0,
    }
