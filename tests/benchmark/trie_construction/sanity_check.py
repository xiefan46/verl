# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Sanity check: validate that V1 and Meituan wrappers produce equivalent
# (per-sample) output for the same input. BLOCKING — benchmark won't run
# if correctness check fails.

"""Output equivalence sanity check between V1 and Meituan wrappers.

Equivalence definition:
  Both wrappers may use different DFS ordering (V1 sorts children by token,
  Meituan preserves input order). What MUST match:
    1. Both find shared prefix of same length (prefix_range)
    2. Both produce leaves for the SAME set of sample indices
    3. For each sample i, the reconstructed tokens (prefix + ancestors + leaf)
       MUST equal the original input sample i, bit-by-bit

Allowable difference:
  - V1 may find DEEPER tree than Meituan (Meituan capped at depth-3). This is
    reported, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SanityResult:
    scenario_name: str
    v1_passed: bool
    meituan_passed: bool
    v1_tree_depth: int  # depth of V1's tree
    mt_tree_depth: int  # depth of Meituan's tree
    notes: list[str]  # human-readable diagnostics

    @property
    def both_passed(self) -> bool:
        return self.v1_passed and self.meituan_passed


def _reconstruct_per_sample(pt_batch, params) -> dict[int, torch.Tensor]:
    """For each sample, reconstruct its full token sequence from the wrapper output.

    Walks q_ranges/k_ranges: for each leaf, find all q==leaf_range, mtype='full' entries
    — those k_ranges are ancestors. Concat ancestors (sorted by start) + leaf range.
    """
    flat = pt_batch.flat_input_ids
    out: dict[int, torch.Tensor] = {}
    for leaf_idx, sample_idx in enumerate(pt_batch.leaf_to_sample):
        leaf_s, leaf_e = pt_batch.leaf_ranges[leaf_idx]
        ancestor_ranges = []
        for (qs, qe), (ks, ke), mtype in zip(params.q_ranges, params.k_ranges, params.mask_types, strict=False):
            if (qs, qe) == (leaf_s, leaf_e) and mtype == "full":
                ancestor_ranges.append((ks, ke))
        all_ranges = sorted(ancestor_ranges) + [(leaf_s, leaf_e)]
        tokens = torch.cat([flat[s:e] for s, e in all_ranges])
        out[sample_idx] = tokens
    return out


def _tree_depth_from_params(params) -> int:
    """Estimate tree depth: max number of ancestor levels for any leaf.

    Walk q_ranges: for each leaf, count distinct k_ranges (full-mtype) it queries —
    that's #ancestors. Tree depth = max #ancestors + 1 (for leaf itself).
    """
    leaf_ancestor_count: dict[tuple[int, int], int] = {}
    for (qs, qe), _, mtype in zip(params.q_ranges, params.k_ranges, params.mask_types, strict=False):
        if mtype == "full":
            leaf_ancestor_count[(qs, qe)] = leaf_ancestor_count.get((qs, qe), 0) + 1
    if not leaf_ancestor_count:
        return 1  # single shared prefix only
    return max(leaf_ancestor_count.values()) + 1


def _check_one(name: str, pt_batch, params, samples: list[torch.Tensor]) -> tuple[bool, list[str]]:
    """Returns (passed, diagnostic notes)."""
    notes = []

    if pt_batch is None:
        return False, [f"{name}: wrapper returned None (no shared prefix detected?)"]

    if params is None:
        return False, [f"{name}: params is None"]

    # Check 1: leaves cover all samples exactly once
    leaf_samples = set(pt_batch.leaf_to_sample)
    expected_samples = set(range(len(samples)))
    if leaf_samples != expected_samples:
        notes.append(f"{name}: leaf_to_sample {sorted(leaf_samples)} != expected {sorted(expected_samples)}")
        return False, notes

    if len(pt_batch.leaf_to_sample) != len(samples):
        notes.append(f"{name}: # leaves {len(pt_batch.leaf_to_sample)} != # samples {len(samples)}")
        return False, notes

    # Check 2: per-sample reconstruction
    reconstructed = _reconstruct_per_sample(pt_batch, params)
    for sample_idx, expected_tokens in enumerate(samples):
        if sample_idx not in reconstructed:
            notes.append(f"{name}: sample {sample_idx} not in reconstructed")
            return False, notes
        actual = reconstructed[sample_idx]
        if actual.numel() != expected_tokens.numel():
            notes.append(f"{name}: sample {sample_idx} len {actual.numel()} != expected {expected_tokens.numel()}")
            return False, notes
        if not torch.equal(actual, expected_tokens):
            # Find first mismatch position for diagnostic
            mismatch = (actual != expected_tokens).nonzero()
            first = int(mismatch[0].item()) if mismatch.numel() > 0 else -1
            notes.append(
                f"{name}: sample {sample_idx} mismatch at pos {first} "
                f"(reconstructed={int(actual[first])}, expected={int(expected_tokens[first])})"
            )
            return False, notes

    notes.append(f"{name}: all {len(samples)} samples reconstructed correctly")
    return True, notes


def check_scenario(scenario, v1_full, v1_params, mt_full, mt_params) -> SanityResult:
    """Check both wrapper outputs against the scenario's input samples."""
    notes: list[str] = []

    v1_ok, v1_notes = _check_one("V1", v1_full, v1_params, scenario.samples)
    notes.extend(v1_notes)

    mt_ok, mt_notes = _check_one("MT", mt_full, mt_params, scenario.samples)
    notes.extend(mt_notes)

    v1_depth = _tree_depth_from_params(v1_params) if v1_params else 0
    mt_depth = _tree_depth_from_params(mt_params) if mt_params else 0

    if v1_ok and mt_ok and v1_depth > mt_depth:
        notes.append(
            f"NOTE: V1 found deeper tree (depth={v1_depth}) than Meituan (depth={mt_depth}). "
            f"Allowable: Meituan capped at depth-3."
        )

    return SanityResult(
        scenario_name=scenario.name,
        v1_passed=v1_ok,
        meituan_passed=mt_ok,
        v1_tree_depth=v1_depth,
        mt_tree_depth=mt_depth,
        notes=notes,
    )


def check_all(scenarios, v1_runner, mt_runner) -> list[SanityResult]:
    """Run all scenarios through both wrappers and return per-scenario sanity results.

    v1_runner / mt_runner: callables (scenario) → (pt_batch, params, timings)
    """
    results = []
    for s in scenarios:
        v1_batch, v1_params, _ = v1_runner(s)
        mt_batch, mt_params, _ = mt_runner(s)
        results.append(check_scenario(s, v1_batch, v1_params, mt_batch, mt_params))
    return results


def print_report(results: list[SanityResult]) -> bool:
    """Print human-readable sanity report. Returns True if all passed."""
    all_passed = True
    print(f"{'Scenario':<25} {'V1':>5} {'MT':>5} {'V1_d':>5} {'MT_d':>5}  Notes")
    print("-" * 100)
    for r in results:
        v1_s = "PASS" if r.v1_passed else "FAIL"
        mt_s = "PASS" if r.meituan_passed else "FAIL"
        marker = "✓" if r.both_passed else "X"
        print(f"{r.scenario_name:<25} {v1_s:>5} {mt_s:>5} {r.v1_tree_depth:>5} {r.mt_tree_depth:>5}  [{marker}]")
        if not r.both_passed:
            all_passed = False
            for n in r.notes:
                if "mismatch" in n.lower() or "fail" in n.lower() or "expected" in n.lower():
                    print(f"  - {n}")
        # Also print depth-divergence notes
        for n in r.notes:
            if n.startswith("NOTE:"):
                print(f"  {n}")
    print()
    if all_passed:
        print("All sanity checks PASSED.")
    else:
        print("SANITY CHECK FAILED — benchmark will NOT proceed.")
    return all_passed


if __name__ == "__main__":
    # Standalone sanity-only run
    from meituan_wrapper import build_prefix_tree_micro_batch_meituan_full
    from scenarios import all_scenarios
    from v1_wrapper import build_prefix_tree_micro_batch_v1_full

    def v1_run(s):
        return build_prefix_tree_micro_batch_v1_full(None, s.samples, prefix_segments_batch=s.prefix_segments_batch)

    def mt_run(s):
        return build_prefix_tree_micro_batch_meituan_full(
            None, s.samples, prefix_segments_batch=s.prefix_segments_batch
        )

    results = check_all(all_scenarios(), v1_run, mt_run)
    ok = print_report(results)
    import sys

    sys.exit(0 if ok else 1)
