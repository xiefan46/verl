# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic data generators for tree training unit tests.

Produces controllable batches where rollouts share a common prompt prefix and
diverge in the response tail — the canonical shape that tree attention is
designed to exploit during RL post-training.
"""

from __future__ import annotations

import torch


def make_prompt_sharing_batch(
    num_prompts: int = 2,
    rollouts_per_prompt: int = 4,
    prompt_len: int = 64,
    response_len: int = 192,
    vocab_size: int = 32000,
    pad_token_id: int = 0,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a synthetic batch where rollouts share a prompt prefix.

    Layout: ``num_prompts`` independent prompts; each prompt has
    ``rollouts_per_prompt`` distinct response tails. All sequences end up the
    same length (``prompt_len + response_len``), so no padding is needed.

    Parameters
    ----------
    num_prompts : int
        Number of distinct prompt groups.
    rollouts_per_prompt : int
        Rollouts sharing each prompt (branching factor of each trie).
    prompt_len : int
        Token count of the shared prompt.
    response_len : int
        Token count of each unique response.
    vocab_size : int
        Vocabulary size used to sample token ids.
    pad_token_id : int
        Token id reserved as padding (excluded from samples so the bool mask
        can be inverted cleanly).
    seed : int
        RNG seed for reproducibility.
    device : str | torch.device
        Device for the returned tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        ``input_ids``      LongTensor ``[B, T]``
        ``attention_mask`` BoolTensor ``[B, T]`` (all True; no padding)
        ``response_mask``  BoolTensor ``[B, T]`` (True on response tokens)
        ``group_ids``      LongTensor ``[B]`` (which prompt group each row is in)

        where ``B = num_prompts * rollouts_per_prompt`` and
        ``T = prompt_len + response_len``.

    Notes
    -----
    Potential overlap ratio (POR) of this batch is
    ``prompt_len / (prompt_len + response_len)``. Vary ``prompt_len`` /
    ``response_len`` to sweep POR in equivalence tests.
    """
    if num_prompts <= 0 or rollouts_per_prompt <= 0:
        raise ValueError("num_prompts and rollouts_per_prompt must be positive")
    if prompt_len <= 0 or response_len <= 0:
        raise ValueError("prompt_len and response_len must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2 (need a non-pad token)")
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError("pad_token_id must lie inside [0, vocab_size)")

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    batch_size = num_prompts * rollouts_per_prompt
    total_len = prompt_len + response_len

    # Sample tokens uniformly from [0, vocab_size) \ {pad_token_id}:
    # draw from [0, vocab_size - 1) then shift ids >= pad_token_id up by one.
    def _sample(shape: tuple[int, ...]) -> torch.Tensor:
        ids = torch.randint(0, vocab_size - 1, shape, generator=generator, device=device)
        return torch.where(ids >= pad_token_id, ids + 1, ids)

    prompts = _sample((num_prompts, prompt_len))
    responses = _sample((batch_size, response_len))

    input_ids = torch.empty(batch_size, total_len, dtype=torch.long, device=device)
    response_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
    group_ids = torch.empty(batch_size, dtype=torch.long, device=device)

    for p in range(num_prompts):
        for r in range(rollouts_per_prompt):
            row = p * rollouts_per_prompt + r
            input_ids[row, :prompt_len] = prompts[p]
            input_ids[row, prompt_len:] = responses[row]
            response_mask[row, prompt_len:] = True
            group_ids[row] = p

    attention_mask = torch.ones(batch_size, total_len, dtype=torch.bool, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "group_ids": group_ids,
    }
