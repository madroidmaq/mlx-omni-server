# mlx-community/Qwen3-1.7B-4bit

from mlx_lm import load, generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
import numpy as np

model_name = "mlx-community/Qwen3-1.7B-4bit"

model, tokenizer = load(model_name)

messages = [{"role": "user", "content": "What is the capital of France? What is the capital of China?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

cache = make_prompt_cache(model)
print(np.shape(cache), cache)
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=256, 
    verbose=True,
    prompt_cache=cache
)
print(np.shape(cache), cache)
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=256, 
    verbose=True,
    prompt_cache=cache
)
print("end")


print(cache[0].keys)


#cache: [layer0, layer 1 .... layer n]
#layer: [batch, num_heads, seq_len, head_dim]

# Save each prompt in LRU dict
# "prompt" : KVCache
# for each prompt, compare common prefix
# retrieve the prompt with highest common prefix and make a copy
# use as prompt cache
# save back after use