# Component-Aware Overlap Reorder

Use this route after option tuning has frozen `Sbest-BASE`, fresh SK-off/SK-on profiling
has classified the winner, and every source-reordered target has an exact
child-to-statement source map. It generalizes the MoE Combine/QBMM experiment without
turning operator names into a universal rule.

## Opportunity Rule

For two dependency-independent statements on reliable distinct streams, prefer the
dispatch order that starts the longer critical resource window first and then launches
shorter complementary-engine work so that the shorter work is covered. The goal is
overlap, not serialization: never add a completion dependency merely to force a visible
order.

The rule is symmetric across Cube and Vector. A typical case is a long Vector window
followed by a short Cube window. If the SK dispatches the Cube work first and then the
Vector work with no overlap, test the dependency-safe permutation `Vector -> Cube`.
Estimate the upper bound from the recoverable overlap, not from the duration of the
longer operator.

Do not generalize literal names such as `MoeDistributeCombineV2` or
`QuantBatchMatmulV3`. Generalize only the measured relationship:

- same exact fused parent and source range;
- stable distinct stream roles;
- complementary resource components;
- no DATA, STREAM_ORDER, event, wait, barrier, communication, cache-mutation,
  side-effect, or control-flow edge forbidding the permutation;
- current dispatch leaves recoverable overlap on the table.

## MIX Component Exception

A MIX business operator remains MIX. It may participate in this route only when a
complete bound short trace exposes all AIC/AIV component intervals for every required
occurrence. Analyze the intended complementary component separately, but also report
contention from every remaining same-engine component. Missing lanes, an overflowed
trace, name-only inference, or an unbound occurrence blocks the trial.

The current `multistream_operator_order.py` v3 pure-resource route must continue to
reject whole-MIX pairs. Do not falsify its `Accelerator Core/Block Num/Mix Block Num`
fields to obtain authorization. A component-aware attempt must preserve a separate
sealed component capture and use the reviewed model adapter/source transform named by
the trial. Run adapter capability preflight with action `component_overlap_reorder`;
until the bound component capture provider, multi-span source mapper, materializer, and
required outputs are available, return `blocked`.

Expose the read-only mapping tree as `artifacts.component_immutable_source_root` and
keep it distinct from the mutable candidate `isolation.source_worktree`. Validate both
the request and result against the immutable tree; the materializer writes only to the
candidate tree.

For the demonstrated MoE pattern, compare the long Combine AIV interval with the QBMM
AIC interval. When the original `QBMM -> Combine` dispatch has zero overlap and the
dependency DAG permits it, test `Combine -> QBMM`; then verify that QBMM AIC becomes
covered by Combine AIV while QBMM AIV contention does not erase the gain.

## Trial And Verification

1. Work in a dedicated source worktree, artifact root, config root, and cache namespace.
2. Ordinary reorders move only complete contiguous statements. A MIX exception may use
   a reviewed cross-function multi-hunk transform when
   `multistream_component_reorder.py validate-map` proves every operator,
   synchronization, join, and transform region is an exact AST span. Preserve child
   set, hard dependencies, events, waits, communication, cache behavior, and the
   single-stream projection.
3. Pass full correctness before performance collection.
4. Verify every target layer/rank keeps the canonical child set, intended dispatch
   order, and at least two reliable streams.
5. Collect a complete bound short trace. Report before/after component P50/P90/MAD,
   overlap, coverage, pair union, parent-SK interval, and the estimated versus realized
   overlap gain.
6. Run three clean processes against the frozen pre-reorder incumbent. Continue to
   exactly five only when mean and median-run directions are strictly positive and tail
   and variance gates pass.
7. Accept after five runs when correctness and evidence bindings remain valid, mean and
   median-run improvements remain strictly positive, and P90/standard deviation do not
   regress. Use `min_improvement_pct=0.0`; there is no fixed 2% floor.
8. Otherwise return `no_gain`, `blocked`, or `failed`, restore the incumbent, and report
   both the local mechanism result and clean end-to-end result.

Local overlap recovery or a faster parent SK is mechanism evidence only. A source
change is selected solely by repeatable clean end-to-end gain.

## Component Artifact Chain

- `superkernel-multistream-component-source-map-v1`: source revision, source-file
  fingerprints, exact AST spans, target/operator bindings, complete hard-dependency
  coverage, and an acyclic union.
- `superkernel-multistream-component-order-capture-v1`: every target range, at least
  three bound occurrences each, stable streams and dispatch order, complete MIX AIC/AIV
  lanes, intended complementary overlap, and remaining same-engine contention.
- `superkernel-multistream-component-source-transform-v1`: reviewed multi-hunk
  replacements confined to transform spans, before/after hunk hashes, unchanged
  single-stream projection and hard-dependency contract, and a post-transform audit.
- `superkernel-multistream-component-dispatch-evidence-v1`: every affected target keeps
  its child/component sets and proves the intended post-transform order before candidate
  profiling starts.

The v1 materializer permits multiple functions in one sealed Python file. A transform
that needs multiple files remains blocked until a later reviewed schema explicitly
supports atomic multi-file replacement.

Invoke materialization with distinct `--output` (candidate source) and
`--manifest-out` (action manifest) paths. Long-option abbreviation is disabled so a
manifest path cannot be mistaken for the candidate source destination.
