# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    # Tree training branch: model_output carries packed-tree tensors that don't
    # go through no_padding_2_padding / TensorDict.select. Dispatch to the
    # tree-aware variant. See research/2026-05-12-tree-training-phase2-design.md §D6.
    if isinstance(model_output, dict) and model_output.get("is_tree_packed"):
        return _ppo_loss_tree(config, model_output, data, dp_group=dp_group)

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def _ppo_loss_tree(config: ActorConfig, model_output: dict, data, dp_group=None):
    """ppo_loss variant for tree-training packed outputs.

    Bypasses ``no_padding_2_padding`` and ``TensorDict.select().to_padded_tensor()``
    because tree micro-batches use plain dicts with flat 1-D tensors in trie
    iteration order. Packed extras (``advantages_packed`` / ``old_log_probs_packed``
    / ``response_mask_packed``) are length ``sum(seq_lens)``; ``log_probs`` from
    :func:`unpack_tree_logprobs` is length ``sum(seq_lens - 1)``. We drop the
    first position of each segment (``align_packed_extras_to_labels``) so all
    four tensors have the same length ``K``, then unsqueeze to ``[1, K]`` and
    reuse the dense ``compute_policy_loss_vanilla`` path unchanged.
    """
    from verl.experimental.tree_training._verl_adapter import (
        align_packed_extras_to_labels,
        segment_lens_from_trie,
    )

    log_prob_flat = model_output["log_probs"]
    trie = model_output["trie"]
    segment_lens = segment_lens_from_trie(trie)

    # Align packed extras to next-token labels: drop position 0 of each segment.
    advantages_flat = align_packed_extras_to_labels(model_output["advantages_packed"], segment_lens)
    old_log_probs_flat = align_packed_extras_to_labels(model_output["old_log_probs_packed"], segment_lens)
    response_mask_flat = align_packed_extras_to_labels(model_output["response_mask_packed"], segment_lens).to(bool)
    ref_log_prob_packed = model_output.get("ref_log_prob_packed")
    ref_log_prob_flat = (
        align_packed_extras_to_labels(ref_log_prob_packed, segment_lens) if ref_log_prob_packed is not None else None
    )

    # global batch info for loss aggregation (mirrors the dense path).
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data.get("global_batch_size")
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data.get("global_batch_size") is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # Promote everything to [1, K] so compute_policy_loss_vanilla's [bsz, response_len]
    # semantics apply unchanged. The trie's all_sequence_ids ordering is
    # canonical and self-consistent across log_probs / advantages / old_log_probs.
    log_prob = log_prob_flat.unsqueeze(0)
    advantages = advantages_flat.unsqueeze(0)
    old_log_prob = old_log_probs_flat.unsqueeze(0)
    response_mask = response_mask_flat.unsqueeze(0)

    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )

    metrics: dict = {}
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # KL loss against ref policy (if enabled). Mirrors the dense path math.
    if config.use_kl_loss and ref_log_prob_flat is not None:
        ref_log_prob = ref_log_prob_flat.unsqueeze(0)
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        # TEMP DEBUG (commit ????????): Phase 4 kl_loss=5.4 bug investigation.
        # ppo_kl=0 after convention-A fix proves log_prob_flat ≈ old_log_probs_packed,
        # but kl_loss=5.4 says log_prob_flat ≠ ref_log_prob_packed. Both should
        # be identical at step 1 (actor==ref weights). Dump samples to localize.
        import os as _os

        if _os.environ.get("VERL_TREE_KL_DEBUG", "0") == "1":
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else 0
            if _rank == 0:
                _resp_mask = response_mask.to(bool)
                _n_resp = _resp_mask.sum().item()
                _resp_log_prob = log_prob[_resp_mask]
                _resp_ref = ref_log_prob[_resp_mask]
                _diff = (_resp_log_prob - _resp_ref).abs()
                _resp_kld = kld[_resp_mask]
                print(
                    f"\n[TREE_KL_DEBUG] response positions: {_n_resp} "
                    f"| log_prob mean={_resp_log_prob.mean().item():.4f} "
                    f"std={_resp_log_prob.std().item():.4f} "
                    f"| ref_log_prob mean={_resp_ref.mean().item():.4f} "
                    f"std={_resp_ref.std().item():.4f} "
                    f"| |diff| mean={_diff.mean().item():.4f} max={_diff.max().item():.4f} "
                    f"| kld mean={_resp_kld.mean().item():.4f}",
                    flush=True,
                )
                # Sample first 8 response positions
                print(
                    f"[TREE_KL_DEBUG] first 8 response positions: "
                    f"log_prob={_resp_log_prob[:8].tolist()} "
                    f"ref={_resp_ref[:8].tolist()}",
                    flush=True,
                )
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss + kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    # Entropy loss is not supported on the tree path in MVP — gather_packed_tree_logprobs
    # returns log_probs only; gather_packed_tree_logprobs_entropy is wired up but the
    # adapter does not surface entropy yet. Validated by the fail-fast: if entropy is
    # in model_output, raise so misuse is loud rather than silent.
    if model_output.get("entropy") is not None:
        raise NotImplementedError(
            "Tree training does not surface entropy in MVP. Wire up unpack_tree_logprobs "
            "via gather_packed_tree_logprobs_entropy to enable entropy regularization."
        )

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
