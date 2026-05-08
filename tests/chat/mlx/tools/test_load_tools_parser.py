from mlx_omni_server.chat.mlx.tools.chat_template import load_tools_parser
from mlx_omni_server.chat.mlx.tools.glm45_tools_parser import GLM45ToolParser
from mlx_omni_server.chat.mlx.tools.hugging_face import HuggingFaceToolParser
from mlx_omni_server.chat.mlx.tools.llama3 import Llama3ToolParser
from mlx_omni_server.chat.mlx.tools.mistral import MistralToolsParser
from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser


class TestLoadToolsParser:
    """Tests for load_tools_parser routing logic."""

    # --- Exact match branches ---

    def test_llama_returns_llama3_parser(self):
        assert isinstance(load_tools_parser("llama"), Llama3ToolParser)

    def test_mistral_returns_mistral_parser(self):
        assert isinstance(load_tools_parser("mistral"), MistralToolsParser)

    def test_qwen2_returns_huggingface_parser(self):
        assert isinstance(load_tools_parser("qwen2"), HuggingFaceToolParser)

    def test_glm4_moe_returns_glm45_parser(self):
        assert isinstance(load_tools_parser("glm4_moe"), GLM45ToolParser)

    # --- Qwen3 prefix branch (all qwen3* variants use XML tool format) ---

    def test_qwen3_returns_qwen3moe_parser(self):
        """qwen3 uses the same XML tool call format as qwen3_moe."""
        assert isinstance(load_tools_parser("qwen3"), Qwen3MoeToolParser)

    def test_qwen3_5_returns_qwen3moe_parser(self):
        """Regression: qwen3_5 model_type must not fall through to default."""
        assert isinstance(load_tools_parser("qwen3_5"), Qwen3MoeToolParser)

    def test_qwen3_moe_returns_qwen3moe_parser(self):
        assert isinstance(load_tools_parser("qwen3_moe"), Qwen3MoeToolParser)

    def test_qwen3_dot_1_moe_returns_qwen3moe_parser(self):
        assert isinstance(load_tools_parser("qwen3.1_moe"), Qwen3MoeToolParser)

    def test_qwen3_dot_5_moe_returns_qwen3moe_parser(self):
        assert isinstance(load_tools_parser("qwen3.5_moe"), Qwen3MoeToolParser)

    def test_qwen3moe_no_separator_returns_qwen3moe_parser(self):
        """qwen3moe also matches startswith('qwen3')."""
        assert isinstance(load_tools_parser("qwen3moe"), Qwen3MoeToolParser)

    # --- Default fallback ---

    def test_unknown_type_returns_huggingface_parser(self):
        assert isinstance(load_tools_parser("unknown"), HuggingFaceToolParser)

    def test_empty_string_returns_huggingface_parser(self):
        assert isinstance(load_tools_parser(""), HuggingFaceToolParser)
