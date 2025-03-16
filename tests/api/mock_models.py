from typing import AsyncGenerator, Dict, List, Optional, Union

import mlx.core as mx
from transformers import PreTrainedTokenizer

from mlx_omni_server.chat.mlx.base_models import BaseTextModel, GenerateResult
from mlx_omni_server.chat.schema import ChatCompletionRequest


class MockLayer:
    """Mock layer for testing"""

    def __init__(self):
        pass

    def __call__(self, x: mx.array, cache=None) -> tuple:
        """Mock forward pass"""
        return x, None


class MockModel(BaseTextModel):
    """Mock MLX model for testing"""

    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.layers = [MockLayer() for _ in range(2)]
        self.vocab_size = 4
        self._tokenizer = MockTokenizer()

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        """Mock forward pass"""
        logits = mx.zeros((x.shape[0], x.shape[1], self.vocab_size))
        # Create a mask for the last token
        last_token_mask = mx.zeros_like(logits)
        last_token_mask = mx.concatenate(
            [last_token_mask[..., :-1], mx.ones_like(last_token_mask[..., -1:])],
            axis=-1,
        )
        logits = logits + last_token_mask  # Add 1.0 to the last token position
        return logits

    def make_cache(self):
        """Mock cache creation"""
        return [None] * len(self.layers)

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[GenerateResult, None]:
        """Generate mock completion text"""
        # Mock a simple response in chunks
        chunks = ["This ", "is ", "a ", "mock ", "response"]
        for i, chunk in enumerate(chunks):
            yield GenerateResult(
                text=chunk,
                token=3,  # Using the "test" token
                finished=i == len(chunks) - 1,
                logprobs=None,
            )

    async def token_count(self, prompt: str) -> int:
        """Count the number of tokens in the text"""
        return len(self._tokenizer.encode(prompt))


class MockTokenizer(PreTrainedTokenizer):
    """Mock tokenizer for testing"""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self._vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "test": 3}
        super().__init__(pad_token="[PAD]")
        self.eos_token_id = 2
        self.pad_token_id = 0

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary"""
        return self._vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        """Mock tokenize method"""
        return ["test"]

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to id"""
        return self._vocab.get(token, 3)  # Default to "test" token

    def _convert_id_to_token(self, index: int) -> str:
        """Convert id to token"""
        for token, idx in self._vocab.items():
            if idx == index:
                return token
        return "test"

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string"""
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence"""
        if token_ids_1 is None:
            return [1] + token_ids_0 + [2]
        return [1] + token_ids_0 + [2] + token_ids_1 + [2]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Get list where entries are [1] if a token is special and [0] else"""
        if already_has_special_tokens:
            return [1 if token in [0, 1, 2] else 0 for token in token_ids_0]

        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs"""
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple:
        """Mock save vocabulary"""
        return ("mock_vocab.txt",)

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Mock chat template application"""
        if tokenize:
            return [1, 2, 3]
        return "Test prompt"
