# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Synthetic data generators for trie construction benchmark.

"""Scenarios for dynamic-trie vs hash-based prefix tree construction benchmark.

Two benchmark families:
  - B1.*: depth-3 scenarios where both dynamic-trie and hash-based work. Fair comparison.
  - B2.*: deep / long scenarios where ONLY the dynamic-trie path works (hash-based hardcoded to depth-3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Scenario:
    name: str
    samples: list[torch.Tensor]
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]]
    expected_depth: int  # for sanity check reporting
    notes: str  # human-readable description


def _hash(tokens: list[int]) -> int:
    return int(hashlib.sha256(b",".join(str(t).encode() for t in tokens)).hexdigest()[:8], 16)


def _gen_tokens(n: int, base: int, rng: torch.Generator) -> list[int]:
    """Generate n distinct random token ids starting around base."""
    return torch.randint(base, base + 10_000, (n,), generator=rng).tolist()


# ============================================================================
# Benchmark 1: depth-3 — both work
# ============================================================================


def _make_depth3_scenario(
    name: str,
    batch_size: int,
    prefix_len: int,
    mid_len: int,
    leaf_len: int,
    num_groups: int,
    seed: int,
    notes: str = "",
) -> Scenario:
    """Generate a depth-3 scenario.

    Tree structure:
      - root prefix: prefix_len tokens (shared by all)
      - num_groups intermediate segments: each mid_len tokens (shared by group)
      - per-sample leaves: leaf_len tokens (unique)

    batch_size must be divisible by num_groups so each group has B/G samples.
    """
    assert batch_size % num_groups == 0
    samples_per_group = batch_size // num_groups
    assert samples_per_group >= 2, "Need ≥2 samples per group for depth-3 (the hash-based path)"

    rng = torch.Generator().manual_seed(seed)

    # Generate shared prefix
    prefix_tokens = _gen_tokens(prefix_len, base=1000, rng=rng)

    samples: list[torch.Tensor] = []
    prefix_segments_batch: list[list[tuple[int, int]]] = []
    sample_idx = 0
    for g in range(num_groups):
        mid_tokens = _gen_tokens(mid_len, base=10000 + g * 100000, rng=rng)
        for _ in range(samples_per_group):
            leaf_tokens = _gen_tokens(leaf_len, base=100000 + sample_idx * 1000, rng=rng)
            full = prefix_tokens + mid_tokens + leaf_tokens
            samples.append(torch.tensor(full, dtype=torch.long))

            # Per-turn segments: turn1=prefix, turn2=prefix+mid, turn3=prefix+mid+leaf
            seg1 = (_hash(prefix_tokens), prefix_len)
            seg2 = (_hash(prefix_tokens + mid_tokens), prefix_len + mid_len)
            seg3 = (_hash(prefix_tokens + mid_tokens + leaf_tokens), prefix_len + mid_len + leaf_len)
            prefix_segments_batch.append([seg1, seg2, seg3])
            sample_idx += 1

    return Scenario(
        name=name, samples=samples, prefix_segments_batch=prefix_segments_batch, expected_depth=3, notes=notes
    )


def b1_scenarios() -> list[Scenario]:
    """Depth-3 fair comparison scenarios."""
    return [
        _make_depth3_scenario(
            "B1.small",
            batch_size=8,
            prefix_len=256,
            mid_len=64,
            leaf_len=64,
            num_groups=2,
            seed=1,
            notes="GRPO-style: B=8, 2 groups, short context (P=256, M=64, L=64)",
        ),
        _make_depth3_scenario(
            "B1.medium",
            batch_size=16,
            prefix_len=512,
            mid_len=128,
            leaf_len=128,
            num_groups=4,
            seed=2,
            notes="Medium: B=16, 4 groups (P=512, M=128, L=128)",
        ),
        _make_depth3_scenario(
            "B1.long_prefix",
            batch_size=8,
            prefix_len=4096,
            mid_len=256,
            leaf_len=256,
            num_groups=2,
            seed=3,
            notes="Long shared prefix: B=8, 2 groups (P=4096, M=256, L=256)",
        ),
        _make_depth3_scenario(
            "B1.long_ctx_8k",
            batch_size=8,
            prefix_len=6000,
            mid_len=1000,
            leaf_len=1000,
            num_groups=2,
            seed=4,
            notes="8K context: B=8, 2 groups (P=6000, M=1000, L=1000)",
        ),
        _make_depth3_scenario(
            "B1.wide_batch",
            batch_size=32,
            prefix_len=512,
            mid_len=128,
            leaf_len=128,
            num_groups=8,
            seed=5,
            notes="Wide batch: B=32, 8 groups (P=512, M=128, L=128)",
        ),
    ]


# ============================================================================
# Benchmark 2: deep trees + long sequences (dynamic only)
# ============================================================================


def _make_deep_scenario(
    name: str,
    batch_size: int,
    depth: int,
    segment_len: int,
    leaf_len: int,
    branch_factor: int,
    seed: int,
    notes: str = "",
) -> Scenario:
    """Generate a deeper-than-3 tree scenario.

    Each level: all samples in a group share segment_len tokens.
    branch_factor: how many subgroups per node.

    Tree shape: root → level2 → ... → level_depth → leaves.
    Total samples = branch_factor^(depth-1) (approximately).
    """
    rng = torch.Generator().manual_seed(seed)

    # Total leaves = batch_size; arrange leaves into a balanced tree
    # depth-N tree with branch_factor B has B^(N-1) leaves at most
    # If batch_size <= B^(N-1), fill leaves greedily

    # Build groups recursively
    def _build_tree(
        level: int, sample_indices: list[int]
    ) -> tuple[list[int], list[tuple[list[int], list[tuple[int, int]]]]]:
        """Returns (shared_segment_tokens, list of (child_sample_idxs, child_seg_history))."""
        seg = _gen_tokens(segment_len, base=1000 + level * 100000, rng=rng)
        return seg, sample_indices

    # Simpler: recursive top-down
    samples_tokens: dict[int, list[int]] = {i: [] for i in range(batch_size)}

    def _grow(level: int, sample_indices: list[int]):
        if level >= depth - 1:
            # Add leaf
            for s in sample_indices:
                leaf = _gen_tokens(leaf_len, base=500000 + s * 10000, rng=rng)
                samples_tokens[s].extend(leaf)
            return
        # Add shared segment for this group
        seg = _gen_tokens(segment_len, base=1000 + level * 100000 + sample_indices[0], rng=rng)
        for s in sample_indices:
            samples_tokens[s].extend(seg)
        # Split into branch_factor subgroups
        n = len(sample_indices)
        if n <= branch_factor:
            for s in sample_indices:
                _grow(level + 1, [s])
        else:
            chunk_size = max(1, n // branch_factor)
            for i in range(0, n, chunk_size):
                _grow(level + 1, sample_indices[i : i + chunk_size])

    _grow(level=0, sample_indices=list(range(batch_size)))

    samples = [torch.tensor(samples_tokens[i], dtype=torch.long) for i in range(batch_size)]

    # Generate prefix_segments_batch — the hash-based path can only see up to depth-3, so segments
    # beyond depth-3 are just leaf tokens. We provide depth-3 segments (turn1=root,
    # turn2=intermediate, turn3=rest) as best-effort for fairness.
    prefix_segments_batch: list[list[tuple[int, int]]] = []
    for s in samples:
        toks = s.tolist()
        # Approximate 3 turns based on segment_len
        t1_end = min(segment_len, len(toks))
        t2_end = min(2 * segment_len, len(toks))
        t3_end = len(toks)
        prefix_segments_batch.append(
            [
                (_hash(toks[:t1_end]), t1_end),
                (_hash(toks[:t2_end]), t2_end),
                (_hash(toks[:t3_end]), t3_end),
            ]
        )

    return Scenario(
        name=name, samples=samples, prefix_segments_batch=prefix_segments_batch, expected_depth=depth, notes=notes
    )


def _make_long_seq_scenario(
    name: str,
    batch_size: int,
    prefix_len: int,
    leaf_len: int,
    seed: int,
    notes: str = "",
) -> Scenario:
    """Single-level (depth-2): N samples share prefix, diverge at leaf.

    For testing long-sequence scaling (e.g., 128K or 1M tokens).
    """
    rng = torch.Generator().manual_seed(seed)
    prefix_tokens = _gen_tokens(prefix_len, base=1000, rng=rng)
    samples = []
    prefix_segments_batch = []
    for i in range(batch_size):
        leaf = _gen_tokens(leaf_len, base=500000 + i * 10000, rng=rng)
        full = prefix_tokens + leaf
        samples.append(torch.tensor(full, dtype=torch.long))
        prefix_segments_batch.append(
            [
                (_hash(prefix_tokens), prefix_len),
                (_hash(full), len(full)),
            ]
        )
    return Scenario(
        name=name, samples=samples, prefix_segments_batch=prefix_segments_batch, expected_depth=2, notes=notes
    )


def b2_scenarios() -> list[Scenario]:
    """Deep trees + long sequences (dynamic only; the hash-based path falls back).

    For "real" depth-D, we need B = branch_factor^(D-1) samples to avoid the dynamic-trie path
    compressing single-sample chains at the bottom of the tree.
    """
    return [
        # --- Real depth scenarios ---
        # depth=4 needs B=8 (= 2^3)
        _make_deep_scenario(
            "B2.depth4_B8",
            batch_size=8,
            depth=4,
            segment_len=128,
            leaf_len=128,
            branch_factor=2,
            seed=11,
            notes="Real depth-4: B=8, branch=2, seg_len=128",
        ),
        # depth=5 needs B=16 (= 2^4)
        _make_deep_scenario(
            "B2.depth5_B16",
            batch_size=16,
            depth=5,
            segment_len=128,
            leaf_len=128,
            branch_factor=2,
            seed=12,
            notes="Real depth-5: B=16, branch=2, seg_len=128",
        ),
        # depth=7 needs B=64 (= 2^6)
        _make_deep_scenario(
            "B2.depth7_B64",
            batch_size=64,
            depth=7,
            segment_len=64,
            leaf_len=64,
            branch_factor=2,
            seed=13,
            notes="Real depth-7: B=64, branch=2, seg_len=64",
        ),
        # depth=8 needs B=128 (= 2^7), tight but feasible
        _make_deep_scenario(
            "B2.depth8_B128",
            batch_size=128,
            depth=8,
            segment_len=32,
            leaf_len=32,
            branch_factor=2,
            seed=14,
            notes="Real depth-8: B=128, branch=2, seg_len=32",
        ),
        # --- Long sequence scenarios ---
        _make_long_seq_scenario(
            "B2.long_64k_B8",
            batch_size=8,
            prefix_len=60_000,
            leaf_len=4_000,
            seed=21,
            notes="64K context (P=60K, L=4K): depth-2 long-seq",
        ),
        _make_long_seq_scenario(
            "B2.long_128k_B4",
            batch_size=4,
            prefix_len=120_000,
            leaf_len=8_000,
            seed=22,
            notes="128K context (P=120K, L=8K)",
        ),
        _make_long_seq_scenario(
            "B2.long_256k_B4",
            batch_size=4,
            prefix_len=240_000,
            leaf_len=16_000,
            seed=23,
            notes="256K context (P=240K, L=16K)",
        ),
        _make_long_seq_scenario(
            "B2.long_1M_B2",
            batch_size=2,
            prefix_len=950_000,
            leaf_len=50_000,
            seed=24,
            notes="1M context (P=950K, L=50K)",
        ),
        # --- Combined deep + long (the killer scenario) ---
        # B=16, depth=5, each path = 4*5000 + 5000 = 25K, total = 400K tokens
        _make_deep_scenario(
            "B2.deep_long_d5_25k",
            batch_size=16,
            depth=5,
            segment_len=5_000,
            leaf_len=5_000,
            branch_factor=2,
            seed=31,
            notes="Combined: real depth-5, ~25K per sample, B=16 (=400K total)",
        ),
        # B=8, depth=4, segment_len=20K, leaf=20K → each path = 60K + 20K = 80K, total = 640K
        _make_deep_scenario(
            "B2.deep_long_d4_80k",
            batch_size=8,
            depth=4,
            segment_len=20_000,
            leaf_len=20_000,
            branch_factor=2,
            seed=32,
            notes="Combined: real depth-4, ~80K per sample, B=8 (=640K total)",
        ),
        # B=4, depth=3, segment_len=100K, leaf=100K → each path = 200K + 100K = 300K, total = 1.2M
        _make_deep_scenario(
            "B2.deep_long_d3_300k_1M",
            batch_size=4,
            depth=3,
            segment_len=100_000,
            leaf_len=100_000,
            branch_factor=2,
            seed=33,
            notes="Combined: depth-3, ~300K per sample, B=4 (=1.2M total)",
        ),
    ]


# ============================================================================
# Benchmark 3: paper-derived realistic configurations
# ============================================================================


def b3_paper_scenarios() -> list[Scenario]:
    """Scenarios derived from published tree-search/tree-RL paper configurations.

    Sources:
      - TreeRL (arXiv 2506.11902): M=6, T=2, N=2, L=1, B=30 responses/prompt, max_seq=8K, 7B-14B
      - rStar-Math (2501.04519): MCTS depth=16, 8-16 candidates/node, 16 rollouts/problem,
                                 avg 5K tokens (MATH), 1.5B-7B
      - DeepSearch (2509.25454): MCTS depth=64, 8 children/expansion, 256 tokens/node,
                                 max response 16K, B=256 global, 1.5B
    """
    return [
        # B3.TreeRL: 1 prompt × 30 rollouts, depth-3 from M=6 parallel sub-trees
        # Tree: prompt (2.5K) → 6 sub-trees (2.5K each, 5 leaves each) → leaves (3K each)
        # per-sample = 2.5K + 2.5K + 3K = 8K
        _make_deep_scenario(
            "B3.TreeRL_B30",
            batch_size=30,
            depth=3,
            segment_len=2500,
            leaf_len=3000,
            branch_factor=6,
            seed=41,
            notes="TreeRL (2506.11902): 1 prompt × 30 rollouts, 6 sub-trees × 5 leaves, per-sample 8K, total 240K",
        ),
        # B3.rStarMath: 16 rollouts at depth-5 sampled from MCTS (full MCTS goes to 16)
        # Each step ~1K tokens; per-sample = 5 × 1K = 5K (MATH avg)
        _make_deep_scenario(
            "B3.rStarMath_B16_d5",
            batch_size=16,
            depth=5,
            segment_len=1000,
            leaf_len=1000,
            branch_factor=2,
            seed=42,
            notes="rStar-Math (2501.04519): 16 rollouts, real depth-5 (sampled from "
            "MCTS depth-16), per-sample 5K, total 80K",
        ),
        # B3.DeepSearch_chain: 8 siblings at a deep MCTS expansion share long chain
        # 15K chain ≈ 60 expansion steps × 256 tokens, + 1K leaf per sibling
        _make_long_seq_scenario(
            "B3.DeepSearch_chain_B8",
            batch_size=8,
            prefix_len=15000,
            leaf_len=1000,
            seed=43,
            notes="DeepSearch (2509.25454): 8 siblings at MCTS leaf, 15K shared chain "
            "(~60 expansions × 256 tok), per-sample 16K, total 128K",
        ),
        # B3.DeepSearch_d5_B16: sub-batch with real depth-5 MCTS tree, per-sample 16K
        # Each level 2K shared + 8K leaf = per-sample 16K
        _make_deep_scenario(
            "B3.DeepSearch_d5_B16",
            batch_size=16,
            depth=5,
            segment_len=2000,
            leaf_len=8000,
            branch_factor=2,
            seed=44,
            notes="DeepSearch-style: B=16 with 5-level shared MCTS tree, per-sample 16K, total 256K",
        ),
    ]


# ============================================================================
# Aggregate
# ============================================================================


def all_scenarios(
    include_b1: bool = True,
    include_b2: bool = True,
    include_b3: bool = True,
) -> list[Scenario]:
    out = []
    if include_b1:
        out.extend(b1_scenarios())
    if include_b2:
        out.extend(b2_scenarios())
    if include_b3:
        out.extend(b3_paper_scenarios())
    return out


if __name__ == "__main__":
    # Smoke test: generate and report sizes for each scenario
    print(f"{'name':<30} {'B':>4} {'depth':>6} {'avg_seq':>8} {'total':>10}  notes")
    print("-" * 100)
    for s in all_scenarios():
        avg = sum(t.numel() for t in s.samples) / len(s.samples)
        total = sum(t.numel() for t in s.samples)
        print(f"{s.name:<30} {len(s.samples):>4} {s.expected_depth:>6} {avg:>8.0f} {total:>10}  {s.notes}")
