from typing import Any, Dict, List, Tuple, Union

import mlx.core as mx
import numpy as np
import tiktoken
from mlx_embeddings import generate, load

from ..utils.logger import logger
from .schema import EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage


class EmbeddingsService:
    """Service for generating embeddings using MLX models (focused on BERT-like models)"""

    def __init__(self):
        # Map of loaded models for caching
        self._models: Dict[str, Tuple[Any, Any]] = {}
        # Default encoder for token counting
        try:
            self._default_tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to another common encoding if cl100k_base is not available
            try:
                self._default_tokenizer = tiktoken.get_encoding("p50k_base")
            except:
                logger.warning(
                    "Could not load any tiktoken encoding, token counts may be inaccurate"
                )
                self._default_tokenizer = None

    def _get_model(self, model_id: str) -> Tuple[Any, Any]:
        """Get or load a model based on its ID"""
        if model_id not in self._models:
            logger.info(f"Loading embedding model: {model_id}")
            try:
                model, processor = load(model_id)
                self._models[model_id] = (model, processor)
            except Exception as e:
                logger.error(f"Error loading embedding model {model_id}: {str(e)}")
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")

        return self._models[model_id]

    def _count_tokens(self, text: Union[str, List[str]]) -> int:
        """Count tokens in input text"""
        if self._default_tokenizer is None:
            # If no tokenizer is available, use a simple approximation
            if isinstance(text, str):
                return len(text.split())
            elif isinstance(text, list):
                return sum(len(t.split()) for t in text)
            return 0

        try:
            if isinstance(text, str):
                return len(self._default_tokenizer.encode(text))
            elif isinstance(text, list):
                return sum(len(self._default_tokenizer.encode(t)) for t in text)
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}. Using fallback method.")
            # Fallback to simple approximation
            if isinstance(text, str):
                return len(text.split())
            elif isinstance(text, list):
                return sum(len(t.split()) for t in text)
        return 0

    def _ensure_float_list(self, embedding) -> List[float]:
        """Ensure embedding is a flat list of float values"""
        if isinstance(embedding, list):
            # Handle case where first element is itself a list or array
            if len(embedding) > 0 and (
                isinstance(embedding[0], list)
                or isinstance(embedding[0], mx.array)
                or isinstance(embedding[0], np.ndarray)
            ):
                return [float(x) for x in embedding[0]]
            # Otherwise, convert each element to float
            return [float(x) for x in embedding]
        elif isinstance(embedding, mx.array):
            # Ensure array is 1D
            if embedding.ndim > 1:
                embedding = embedding.reshape(-1)
            return [float(x) for x in embedding.tolist()]
        elif isinstance(embedding, np.ndarray):
            # Ensure array is 1D
            if embedding.ndim > 1:
                embedding = embedding.reshape(-1)
            return [float(x) for x in embedding.tolist()]
        else:
            # Handle any other unexpected type
            return [float(x) for x in list(embedding)]

    def _get_bert_embeddings(self, model, processor, text, model_id):
        """Extract embeddings specifically for BERT-like models"""
        # Use proper encode method - processor may have different method names
        encode_method = getattr(processor, "encode", None)
        if encode_method is None:
            encode_method = getattr(processor, "batch_encode_plus", None)

        if encode_method:
            # Use encode or batch_encode_plus
            input_ids = encode_method(text, return_tensors="mlx")

            # Handle different input formats
            if isinstance(input_ids, dict):
                outputs = model(**input_ids)
            else:
                outputs = model(input_ids)

            # Extract embeddings based on model output structure
            if hasattr(outputs, "last_hidden_state"):
                # For models like BERT, MiniLM
                # Check if model is likely MiniLM (which typically uses CLS token)
                if "minilm" in model_id.lower():
                    # MiniLM models typically use the CLS token (first token)
                    return outputs.last_hidden_state[:, 0, :]
                else:
                    # Other BERT models might use mean pooling
                    return outputs.last_hidden_state.mean(axis=1)
            else:
                # Last resort, assume the output itself is the embedding
                return outputs

        raise ValueError(f"Could not determine how to extract embeddings from model")

    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings based on the request"""
        model_id = request.model
        model, processor = self._get_model(model_id)

        # Handle both string and list of strings
        inputs = request.input if isinstance(request.input, list) else [request.input]

        # Count tokens for usage info
        token_count = self._count_tokens(inputs)

        # Generate embeddings for all inputs
        embeddings = []
        for idx, text in enumerate(inputs):
            try:
                # Generate embedding using the model
                try:
                    # First try the specific BERT extraction method
                    embedding = self._get_bert_embeddings(
                        model, processor, text, model_id
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed with BERT method: {str(e)}. Trying general generate() function."
                    )
                    # Fall back to the generate function
                    output = generate(model, processor, text)
                    if hasattr(output, "last_hidden_state"):
                        embedding = output.last_hidden_state[:, 0, :]
                    else:
                        embedding = output

                # Convert to list of floats with proper formatting
                embedding_list = self._ensure_float_list(embedding)

                # Create embedding data
                embedding_data = EmbeddingData(
                    embedding=embedding_list,
                    index=idx,
                )
                embeddings.append(embedding_data)

            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to generate embedding: {str(e)}")

        # Create the full response
        response = EmbeddingResponse(
            data=embeddings,
            model=model_id,
            usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count),
        )

        return response
