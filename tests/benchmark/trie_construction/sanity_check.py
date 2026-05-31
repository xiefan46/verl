# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Sanity check: validate that dynamic-trie and hash-based wrappers produce equivalent
# (per-sample) output for the same input. BLOCKING — benchmark won't run
# if correctness check fails.

"""Output equivalence sanity check between dynamic-trie and hash-based wrappers.

Equivalence definition:
  Both wrappers may use different DFS ordering (dynamic-trie sorts children by token,
  hash-based preserves input order). What MUST match:
    1. Both find shared prefix of same length (prefix_range)
    2. Both produce leaves for the SAME set of sample indices
    3. For each sample i, the reconstructed tokens (prefix + ancestors + leaf)
       MUST equal the original input sample i, bit-by-bit

Allowable difference:
  - the dynamic-trie path may find a DEEPER tree than the hash-based path (hash-based capped at depth-3). This is
    reported, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SanityResult:
    scenario_name: str
    dyn_passed: bool
    hash_passed: bool
    dyn_tree_depth: int  # depth of the dynamic-trie's output
    mt_tree_depth: int  # depth of hash-based's tree
    notes: list[str]  # human-readable diagnostics

    @property
    def both_passed(self) -> bool:
        return self.dyn_passed and self.hash_passed


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


def check_scenario(scenario, dyn_full, dyn_params, mt_full, mt_params) -> SanityResult:
    """Check both wrapper outputs against the scenario's input samples."""
    notes: list[str] = []

    dyn_ok, dyn_notes = _check_one("dynamic", dyn_full, dyn_params, scenario.samples)
    notes.extend(dyn_notes)

    mt_ok, mt_notes = _check_one("MT", mt_full, mt_params, scenario.samples)
    notes.extend(mt_notes)

    dyn_depth = _tree_depth_from_params(dyn_params) if dyn_params else 0
    mt_depth = _tree_depth_from_params(mt_params) if mt_params else 0

    if dyn_ok and mt_ok and dyn_depth > mt_depth:
        notes.append(
            f"NOTE: dynamic-trie found deeper tree (depth={dyn_depth}) than hash-based (depth={mt_depth}). "
            f"Allowable: hash-based capped at depth-3."
        )

    return SanityResult(
        scenario_name=scenario.name,
        dyn_passed=dyn_ok,
        hash_passed=mt_ok,
        dyn_tree_depth=dyn_depth,
        mt_tree_depth=mt_depth,
        notes=notes,
    )


def check_all(scenarios, dyn_runner, mt_runner) -> list[SanityResult]:
    """Run all scenarios through both wrappers and return per-scenario sanity results.

    dyn_runner / mt_runner: callables (scenario) → (pt_batch, params, timings)
    """
    results = []
    for s in scenarios:
        dyn_batch, dyn_params, _ = dyn_runner(s)
        mt_batch, mt_params, _ = mt_runner(s)
        results.append(check_scenario(s, dyn_batch, dyn_params, mt_batch, mt_params))
    return results


def print_report(results: list[SanityResult]) -> bool:
    """Print human-readable sanity report. Returns True if all passed."""
    all_passed = True
    print(f"{'Scenario':<25} {'dynamic':>5} {'MT':>5} {'Dyn_d':>5} {'MT_d':>5}  Notes")
    print("-" * 100)
    for r in results:
        dyn_s = "PASS" if r.dyn_passed else "FAIL"
        mt_s = "PASS" if r.hash_passed else "FAIL"
        marker = "✓" if r.both_passed else "X"
        print(f"{r.scenario_name:<25} {dyn_s:>5} {mt_s:>5} {r.dyn_tree_depth:>5} {r.mt_tree_depth:>5}  [{marker}]")
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
    from dynamic_trie_wrapper import build_prefix_tree_micro_batch_dynamic_full
    from hash_based_wrapper import build_prefix_tree_micro_batch_hash_based_full
    from scenarios import all_scenarios

    def dyn_run(s):
        return build_prefix_tree_micro_batch_dynamic_full(
            None, s.samples, prefix_segments_batch=s.prefix_segments_batch
        )

    def mt_run(s):
        return build_prefix_tree_micro_batch_hash_based_full(
            None, s.samples, prefix_segments_batch=s.prefix_segments_batch
        )

    results = check_all(all_scenarios(), dyn_run, mt_run)
    ok = print_report(results)
    import sys

    sys.exit(0 if ok else 1)
