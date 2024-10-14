import os
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(api_key=os.environ["model_api_key"],base_url=os.environ["model_base_url"])
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content

async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.environ["llm_model_name"],
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.environ["llm_model_name"],
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    embed_model = HuggingFaceEmbedding(model_name=os.environ["embed_model_name"],)
    response = embed_model.get_text_embedding_batch(texts)
    return np.array([dp for dp in response])

if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete('How are you?')
        print(result)

    asyncio.run(main())