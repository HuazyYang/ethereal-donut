# RDG Design: Production Plan over the Existing FrameGraph

- Status: Accepted
- Date: 2026-10-12
- Scope: Ethereal Donut pass composition and execution over NVRHI
- Related: aggregate plan [2026-10-04-framegraph.md](../../docs/plans/2026-10-04-framegraph.md);
  donut ADRs 0001-0007 and nvrhi ADRs 0008-0009 on the `framegraph` branches

The Render Dependency Graph (RDG) is the subsystem in which passes declare what they read and write
and the framework derives execution order, resource allocation, barriers, queue synchronisation and
introspection. RDG, render graph and frame graph are synonyms here; in code the subsystem is
`donut::render::framegraph`.

An earlier draft of this document (2026-10-12, withdrawn) proposed evaluating AMD Render Pipeline
Shaders (RPS) as the graph compiler and runtime before committing to a compiler over NVRHI. Section 1
settles that question from the SDK headers and from this repository, without a prototype. Sections 2
to 4 are the production plan for the implementation that exists.

## 1. Goals and the compiler decision

### 1.1 Production goals

- Declare pass dependencies and resource accesses explicitly.
- Validate dependencies before command recording and cull unused work.
- Manage transient resources, persistent history and external resources correctly across frames in flight.
- Generate correct resource transitions and queue dependencies for D3D12 and Vulkan.
- Support parallel command recording and measured asynchronous compute.
- Explain execution order, memory usage, barriers and queue waits through diagnostics.
- Migrate existing passes incrementally while preserving a comparison path.

C++ is the authoring language. Replacing NVRHI, a scripting language and a visual graph editor are out
of scope. D3D11 runs with reduced capability (serial execution, pooling, no placed resources).

### 1.2 The `framegraph` branches

A complete custom RDG exists, unmerged, on `framegraph` branches in all four repositories. Measured
against the checked-out tips on 2026-10-12:

| Repository | Ahead of tip | Behind tip | What is behind |
| --- | --- | --- | --- |
| ethereal-nvrhi (`ethereal-dev`) | 11 | 3 | EPM package manager, test header targets |
| ethereal-donut (`ethereal-dev`) | 25 | 7 | EPM, `query_cast`/interface tables, scene-graph child list, build fixes |
| ethereal-samples (`main`) | 6 | 12 | nv_asteroids, EPM, media fetch, `query_cast` |
| aggregate (`main`) | 10 | 12 | presets, vcpkg, EPM, nv_asteroids option |

What the branch contains, from its own records:

| Part | Lines | State |
| --- | --- | --- |
| Core compiler and executor (`src/render/framegraph`, public headers) | ~12,200 | Builder, SSA versions, channels, culling, validation, lifetimes, Tier 0 pooling, Tier 1 placed aliasing, segment scheduling across queues, diagnostics, inspector, ImGui visualizer |
| Device-free logic (`detail/`) | ~1,450 | Topological sort, interval lifetimes, placement solver, pool |
| Pass adapters (11 passes) | ~4,150 | GBuffer, deferred lighting, depth, forward, sky, mipmap, SSAO, bloom (5 graph passes), tone mapping, TAA (2 graph passes), blit |
| Tests | ~5,300 | 7 executables plus ABI lint and header hygiene; 18/18 pass in Debug and Release on D3D12, Vulkan and D3D11 under the NVRHI validation layer, Vulkan validation layer and D3D12 debug runtime |
| nvrhi fork | | `ICommandList1::aliasingBarrier`, `Feature::AliasingBarrier`, `Shared_CrossQueue`, Vulkan bind-result fix, D3D12 legacy aliasing transition fix |

Three samples run under the graph (`deferred_shading_fg`, `variable_shading_fg`, `async_compute_fg`)
with zero validation messages over 120 Debug frames; `deferred_shading_fg` screenshots are
byte-identical between D3D12 and D3D11; `variable_shading_fg` matches the original's 42 draws. The work
also found ten pre-existing defects in donut, nvrhi and the samples, including a unit-test suite that
had never run.

Of the goals in 1.1, every one except parallel recording has code and tests on this branch. The
remaining gaps are listed in section 3.

### 1.3 The RPS custom-runtime route is a fork, not an adapter

The decisive question for RPS is whether a custom runtime can express NVRHI transition and aliasing
semantics without an invasive fork. The public header answers it. `RpsRuntimeCallbacks` in
`rps_runtime_callbacks.h` (main, 2024) contains exactly:

```
pfnBuildRenderGraphPhases, pfnDestroyRuntime,
pfnCreateHeap, pfnDestroyHeap, pfnCreateResource, pfnDestroyResource,
pfnCreateNodeResources, pfnDestroyNodeResources,
pfnRecordDebugMarker, pfnSetDebugName
```

There is no command-recording callback and no barrier or transition callback. Those live in the C++
`RuntimeBackend` phases under `src/runtime/common/`, which the D3D12 and Vulkan backends subclass, and
which a custom runtime reaches only through `pfnBuildRenderGraphPhases` returning internal phase
objects. An NVRHI runtime therefore compiles against RPS internals and tracks their changes.

Two further points from the headers and the repository page:

- The builder API without RPSL does exist: `RpsRenderGraphUpdateInfo::pfnBuildCallback`,
  `rpsRenderGraphDeclareResource`, `rpsRenderGraphAddNode`, `rpsRenderGraphAllocateData`. A C++
  frontend over RPS would have been possible; the backend is what fails.
- Licensing is MIT in the headers and `LICENSE.txt`, while the README still carries evaluation-license
  text. Upstream has 7 commits on `main`, the last on 2024-05-14 (Open Beta 1.1.1), no releases, 11 open
  issues and 5 open pull requests. Upstream maintenance would be ours.

### 1.4 What RPS would and would not replace

| RPS would replace | Size here | RPS would not replace |
| --- | --- | --- |
| Topological sort, lifetime intervals, placement solver | ~1,450 lines, the most heavily tested code on the branch | The nvrhi aliasing barrier and cross-queue sharing fixes |
| Scheduling heuristics (program order, memory saving, compute/graphics switch minimisation, split barriers, frame overlap) | Branch has reordering but fewer heuristics | Framebuffer cache keyed on the whole descriptor, import state contract, state restore after opaque passes |
| | | All eleven pass adapters and the additive pass refactors |
| | | Diagnostics, inspector, visualizer, debug switches, determinism options |
| | | A second resource-state tracker beside NVRHI's, which RPS would add |

The replaceable part is small and already verified; the unreplaceable part is where the branch's
defects were actually found. RPS's scheduler heuristics are the one thing worth borrowing, and they
are borrowed as a checklist (`RpsScheduleFlags`), not as a dependency.

### 1.5 Other implementations consulted

Reference material only. None is a candidate dependency.

| Implementation | Useful for | Why not adopted |
| --- | --- | --- |
| [Falcor](https://github.com/NVIDIAGameWorks/Falcor) | Named pass inputs and outputs, reflection, access to intermediates | Its executor and resource cache have no multi-queue scheduler or lifetime-based allocator |
| [Granite](https://github.com/Themaister/Granite) | Vulkan synchronisation, async compute, history, aliasing details | Coupled to Granite's Vulkan abstraction; unsupported |
| [Unreal RDG](https://dev.epicgames.com/documentation/unreal-engine/render-dependency-graph-in-unreal-engine) | Subresource transitions, transient allocation, parallel recording, RDG Insights | Not separable from Unreal |
| [Filament FrameGraph](https://google.github.io/filament/notes/framegraph.html) | C++ authoring model, culling, graph viewer | Embedded in Filament |
| [O3DE Atom](https://www.docs.o3de.org/docs/api/gems/atom/class_a_z_1_1_r_h_i_1_1_frame_graph.html) | Compile/execute separation inside an RHI | Carries the engine's RHI |
| [Unity RenderGraph](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/render-graph-introduction.html) | Authoring and migration patterns | C# |
| [Daxa TaskGraph](https://github.com/Ipotrick/Daxa) | Explicit task accesses, compiled graph reuse, bindless | Would replace the GPU abstraction |

### 1.6 Constraints already settled on the branch

- Lifetime classes are Transient, Exported, Imported, Persistent and Culled (donut ADR 0002).
- Pooling (Tier 0) came before placed aliasing (Tier 1), and Tier 1 required the nvrhi aliasing barrier.
- One transient heap per queue; a resource touched by more than one queue is never aliased
  (ADR 0005, enforced by `detail::canReuse`).
- D3D11 runs Tier 0 on one queue; `deferred_shading_fg` is the D3D11 sample.
- Vulkan Tier 1 is blocked on `createHeap` ignoring `memoryTypeBits` (nvrhi follow-up 2); the
  discarded bind result is fixed on the fork.
- A host texture created with `keepInitialState` (the swap chain) must not have its state declared to
  the graph; doing so cost the device once.

### 1.7 Decision

The RDG is the `framegraph` branch. RPS is not adopted and no prototype is built; donut ADR 0010
records the decision with the evidence of 1.3 and 1.4.

## 2. Production architecture

Unchanged from the branch, stated here so this document stands alone:

```
Pass adapters (donut conventions)  |  New graph-native passes
        IFrameGraphPass: declare / refineDeclarations / createPipelines / bindResources / prepare / execute
                                   |
IFrameGraphBuilder: typed ResourceId with SSA versions, ChannelId edges, dense SlotId per pass,
                    render-target groups, queue preference, view iteration
                                   |
Compile: validation -> culling -> order -> lifetimes (half-open, per pass instance) ->
         residency class -> Tier 0 pool / Tier 1 placement -> segments per queue -> immutable plan
                                   |
Execute: per segment one command list; beginTracking from the plan, aliasing barriers,
         setTextureState per declared use, one commitBarriers, pass bodies, restore, fences in order
                                   |
NVRHI (automatic barriers on, pre-transitioned so they find nothing to do)
```

Rules the production version keeps as hard contracts:

1. **Queues are declared, never inferred.** A demotion is a warning.
2. **A cross-queue resource is never aliased.** Fence-aware reuse is a later change to `canReuse` only.
3. **Persistent resources are never pooled or aliased** and carry their tracked state across frames.
4. **Culling cannot remove a pass with declared side effects**, and every output the host consumes is a
   marked channel output.
5. **The core public headers obey the nvrhi ABI rules and are linted**; adapters follow donut
   conventions. This is enforced today (9 interfaces, 147 virtuals, 25 classes) and stays enforced.

## 3. What production requires beyond the branch

### 3.1 Land it (Phase 0)

Merge or rebase in the forced order nvrhi -> donut -> samples -> aggregate. Expected conflict areas,
from the commits each branch is behind:

- Third-party resolution: the branch's `thirdparty/CMakeLists.txt` and `imgui.cmake` edits versus EPM.
- `query_cast` and explicit interface tables (donut `5d94818`): every `ObjectImpl` class the branch
  adds needs an interface table; the adapters' `IGeometryPass` use must match the new tables.
- Scene-graph child list (donut `eef107a`): the geometry adapters call
  `drawStrategy.PrepareForView(root, view)` per view instance; check the root type.
- `build.ps1` and presets: the branch predates the vcpkg triplet and submodule check fixes.

Then default `DONUT_WITH_UNIT_TESTS` to `ON` in the aggregate (branch follow-up 9). A test count of 18
is the pass criterion; fewer is a failure.

### 3.2 Correctness gate (Phase 1)

- Deterministic A/B on the three ported samples against their originals, on D3D12, Vulkan and D3D11:
  same draw count, bit-identical back buffer with `-fixedstep` added to the originals, per-intermediate
  comparison, and the RenderDoc interval check on every aliasing pair. This is branch follow-up 1 and
  the only correctness item still unverified.
- The aliasing negative test: `ForceCommitted` versus Tier 1 on the same frame must match, and removing
  one `aliasingBarrier` must make them differ. Without it Tier 1 is "accepted by the API", not proven.
- One D3D12 GPU-based-validation run per sample.
- ADR 0010: RPS not adopted, with the header evidence of 1.3 and the upstream data of 1.3.

### 3.3 Contracts to harden (Phase 2)

These are the places where the branch record shows the design was discovered by debugging rather than
stated up front. Each becomes a declared, validated contract.

**Import kinds.** Replace the optional initial state on `ImportedTextureDesc` with an explicit kind:

```cpp
enum class ImportKind : uint8_t
{
    HostTracked,    // created with keepInitialState; the graph never declares its state (swap chain)
    StateDeclared,  // graph declares entryState and exitState; host guarantees entryState on arrival
};
```

Compile error when `HostTracked` is paired with a state, or `StateDeclared` with a texture whose
`getDesc().keepInitialState` is true. This turns the device-loss bug into a diagnostic.

**External-pass boundary.** A pass flag `FrameGraphPassFlags::OpaqueExternal` for bodies the graph cannot
see into (wrapped passes that pick bindings at run time, SDK effects, native API work). For such a pass
the executor commits every declared resource to its declared state before the body and restores after;
for every other pass the after-restore is dropped, which removes a per-pass cost the branch currently
pays everywhere. The restore runs before any memory is handed to an alias, as now.

**History without parity hacks.** `TemporalAntiAliasingPass` fails the ADR 0003 wrap test on condition
1: its seven textures are constructor parameters and it selects the feedback texture by frame parity
behind the graph's back. Refactor it additively like the others: a constructor taking formats, a `Render`
taking textures. The adapter then uses `createHistoryTexture` to get current/history pairs, declares the
history as a read and the current as a write, and passes `feedbackIsValid = !wasSlotReallocated(history)`.
The exposure buffer in `ToneMappingPass` follows the same path.

**Retirement.** A resize or full re-allocation moves replaced persistent and exported resources to a
retirement list keyed by the frame that last used them, released from `execute()` once the device's
lifetime tracker reports that frame complete, through `ICommandListLifetimeTracker::runGarbageCollection`.
The branch holds references through the re-allocation; this gives the rule a name and a test.

**Frame-boundary handoff.** The plan records, per persistent and cross-queue resource, the state it is
left in at the end of the frame and the queue that left it there. The first segment on each queue that
touches such a resource waits for the previous frame's last submission on the other queues, as the branch
does on D3D12; the rule becomes backend-neutral and is covered by the device test on Vulkan as well.

### 3.4 Performance budgets (Phase 3)

Budgets are set before any comparison is run. Proposed numbers, to be confirmed on the first
measurement and then frozen:

| Measure | Budget | Where |
| --- | --- | --- |
| Full compile, <= 64 passes | <= 2 ms on one desktop core | `variable_shading_fg` |
| Rebind-only recompile (new swap-chain image) | <= 0.1 ms | every sample, every frame |
| Executor CPU overhead over the hand-written frame | <= 0.3 ms | `deferred_shading` versus `_fg` |
| GPU frame time at equal draw count | within 1% | all three pairs, NVIDIA and AMD |
| Transient bytes at Tier 0 | <= 60% of the no-reuse baseline | `FrameGraphMemoryStats` |
| Transient bytes at Tier 1 | reported separately; no target until the A/B passes | D3D12 only |

Tier 0 pooling is the production default on all backends. Tier 1 becomes a default only after the
Phase 1 gate passes on two vendors; until then it is opt-in per application. Asynchronous compute is
opt-in per pass and reported as a warning when demoted, as now.

### 3.5 Parallel recording (Phase 4)

The one goal in 1.1 with no code on the branch. A segment is already one command list and one
submission. The design splits a segment into recording ranges at pass boundaries, each its own command
list, recorded on worker threads and executed in order with one `executeCommandLists` call per segment:

```cpp
NVRHI_IID(IFrameGraphRecorder, "...")
struct IFrameGraphRecorder : nvrhi::IObject
{
    // Called once per recording range; the implementation runs it on any thread and must
    // return only when every range has been recorded. Ranges never share a command list.
    virtual FRESULT recordRanges(IFrameGraphRecordRange* const* ranges, uint32_t count) noexcept = 0;
};
```

Preconditions the branch record already establishes:

- Resource states at every pass boundary are known from the plan, so each range opens with its own
  `beginTrackingTextureState` set and recording order is irrelevant to correctness.
- Volatile constant buffers are per command list, so `prepare()` may not write them; only `execute()`
  may. The branch hit this once (`ForwardShadingFGPass`); it becomes a validation-layer check.
- `engine::BindingCache` is guarded by a `std::shared_mutex`, so concurrent recording is safe; whether
  one shared cache contends under many ranges is a measurement, and per-range caches are the fallback.

Enable it only when a segment holds more passes than a measured threshold; the single-list path stays
the default and the differential baseline.

### 3.6 Migration (Phase 5)

Originals stay until Phase 1 passes bit-exact. Then migrate sample by sample, delete the `_fg`
duplicates, port `vxgi_samples` last behind its `-framegraph` switch, and leave the sixteen samples that
never used the graph untouched. nvrhi follow-ups 1 (expose a virtual resource's heap binding, for the
overlap check) and 2 (expose and honour `memoryTypeBits`, for Vulkan Tier 1) are scheduled here unless a
backend defect pulls them forward.

## 4. Phases and exit criteria

| Phase | Delivers | Exit criterion |
| --- | --- | --- |
| 0 Land | `framegraph` branches merged onto `ethereal-dev`/`main`; tests default on | Whole aggregate builds Debug and Release; 18/18 tests |
| 1 Correctness | Deterministic A/B, aliasing negative test, GBV run, ADR 0010 | Bit-identical back buffers on three backends; negative test fails when a barrier is removed |
| 2 Contracts | Import kinds, opaque-pass boundary, TAA/exposure history refactor, retirement, frame handoff | Each contract has a compile-time diagnostic and a device-test case |
| 3 Performance | Budgets measured on NVIDIA and AMD; Tier 1 default decision; async compute opt-in | Every budget in 3.4 met or renegotiated in writing |
| 4 Parallel recording | `IFrameGraphRecorder`, per-range command lists, thread-safety checks | Equal images to the single-list path; CPU record time measured |
| 5 Migration | Originals migrated, duplicates deleted, vxgi ported, nvrhi follow-ups 1-2 | No `_fg` directories remain; Vulkan Tier 1 enabled or explicitly deferred |

## 5. Out of scope

- An RPS prototype or adoption gate; section 1 is the record.
- A separate feasibility step; the branch is the feasibility result.
- D3D11 parity with the explicit APIs; D3D11 is Tier 0, single queue, and tested.
- Adopting any implementation in 1.5; they remain references.

## 6. Open decisions

Three decisions are left to the owner. Each is stated with what is known, the options, when it must be
taken, and a recommendation. None blocks Phase 0 or Phase 1.

### 6.1 Binding-set caching under parallel recording (decide before Phase 4)

**What is known.** `engine::BindingCache` is one shared map guarded by a `std::shared_mutex`. A lookup
takes the shared lock. A miss takes the exclusive lock and calls `createBindingSet` while holding it, so
every thread that misses at the same moment waits for one descriptor allocation at a time. Hits scale;
misses serialise. Misses happen when physical resource identity changes: the first frame after a compile,
after a resize, and whenever pooling rotates a texture into a slot. In steady state a frame should have
no misses; the question is how long the first frame after a change stalls and whether that matters.

**Options.**

| Option | Cost | Risk |
| --- | --- | --- |
| A. Keep the shared cache; measure | None now | A visible hitch on the first frame after a resize if many ranges miss together |
| B. Create under the shared lock, insert under the exclusive lock, drop duplicates | ~20 lines in `BindingCache` | Two threads may create the same binding set once; the loser's is released. Descriptor churn is bounded by thread count |
| C. One `BindingCache` per recording range, cleared with the plan | Adapters take the cache from `IFrameGraphResources`, not from the graph | Duplicates the same binding sets in every range; multiplies descriptor-heap use by the range count |
| D. Pre-warm: `bindResources` creates every binding set the pass will need, `execute` only looks up | Every adapter must know its binding sets ahead of time | Adapters that choose bindings at run time (TAA today, SSAO's indexed sets) cannot pre-warm until refactored under 3.3 |

**Recommendation.** A now, B when Phase 4 starts, because B removes the only serialisation point for a
small change and needs no adapter work. C only if measurements show descriptor-heap pressure is not a
concern on the target hardware. D is the eventual shape once 3.3 has removed run-time binding choices,
and it also makes `execute` allocation-free, which parallel recording wants anyway.

### 6.2 Vulkan Tier 1: how placed resources get the right memory type (decide after Phase 3 numbers)

**What is known.** Vulkan requires a placed image's memory to come from a type in that image's
`memoryTypeBits`. nvrhi's `createHeap` passes `memoryTypeBits = ~0u` and the allocator takes the first
type that matches the property flags, and `MemoryRequirements` carries only `size` and `alignment`, so
the caller cannot check. On most desktop drivers every optimal-tiling image reports the same device-local
type, so this works by accident; the specification does not promise it, and drivers do restrict types for
some depth/stencil and lazily-allocated formats. The branch therefore keeps Vulkan at Tier 0 (pooling),
which is correct and loses only the extra saving aliasing would bring on top of pooling. That saving is
not yet measured anywhere; Phase 3 produces it on D3D12, where Tier 1 already runs.

**Options.**

| Option | What changes | Size | Consequence |
| --- | --- | --- | --- |
| A. Expose the mask | Add `memoryTypeBits` to `MemoryRequirements` and a type mask to `HeapDesc`; the Vulkan `createHeap` honours it; the graph groups resources by compatible mask and creates one heap per group | Two struct fields, one `c_HeaderVersion` bump (currently 28), ~50 lines in the Vulkan backend, a grouping step in the placement solver | Keeps the explicit heap-and-offset model the D3D12 path uses; every consumer recompiles for the header bump |
| B. Allocator inside nvrhi (VMA-style) | The Vulkan backend owns a sub-allocator; `createHeap` becomes a pool with a requested mask union, and placement is checked at `bindTextureMemory` | A new dependency or ~1,000 lines, plus a bind-time failure path the graph must handle | Hides the problem in the backend but adds a second allocator under the graph's own placement solver, which already decides offsets; the two will disagree |
| C. Stay at Tier 0 on Vulkan | Nothing | None | Vulkan keeps pooling only. Acceptable if Phase 3 shows Tier 1 saves little over Tier 0 on real scenes |
| D. Probe at start-up | Query `memoryTypeBits` for one image per format class the graph uses; enable Tier 1 only if all masks are identical | Needs A's first field, no `HeapDesc` change | Correct on hardware where the accident holds, Tier 0 elsewhere; a half-step towards A |

**Recommendation.** Do not decide until Phase 3 has the D3D12 number for aliased bytes over pooled
bytes. If the extra saving is under about 15% of transient memory on `variable_shading_fg`, take C and
close the item. Otherwise take A: it is the smallest correct change, it stays inside the ABI rules, and
it needs nvrhi follow-up 2 in any case, which is the same header bump. B is not recommended: the graph
already owns placement, and a second allocator beneath it is the design the branch avoided on D3D12.

### 6.3 Ownership of the nvrhi follow-ups and the GBV run (decide at Phase 0)

**What is known.** GBV is D3D12 GPU-Based Validation: a debug-layer mode that instruments shaders to
check resource states, descriptors and out-of-bounds access at the point of use on the GPU. It is the
only check that catches a resource used in the wrong state at dispatch time, which is exactly the class
of error an aliasing or barrier bug produces. It runs 10 to 100 times slower than the debug layer alone,
needs the Graphics Tools optional Windows feature, and donut already wires it:
`DeviceCreationParameters::enableGPUValidation` reaches `SetEnableGPUBasedValidation` in the D3D12
device manager. No sample or test sets it today; the ported samples set `enableDebugRuntime` only, so
GBV has never been run on the graph.

The two nvrhi follow-ups are changes to the fork's public header, each a `c_HeaderVersion` bump:

1. Expose a virtual resource's heap and offset, so the validation layer can check that two resources
   named in an `aliasingBarrier` genuinely overlap. Nothing else can catch a wrong offset.
2. Expose and honour `memoryTypeBits` (the first field of option A in 6.2).

**The decision is sequencing, not people.** This is one owner's fork, so "who" reduces to "when" and
"how often".

| Item | Early (Phase 0 or 1) | Late (Phase 5) |
| --- | --- | --- |
| Follow-up 1, heap binding | The Phase 1 A/B runs with the overlap check on, so a wrong placement is a validation error rather than a plausible image | Phase 1 relies on image comparison alone for aliasing correctness |
| Follow-up 2, memoryTypeBits | Shares the header bump with follow-up 1; one recompile of everything instead of two | Only needed if 6.2 chooses A or D; may never be needed |
| GBV run | Once per ported sample before the Phase 1 gate, by hand, with a `-gpuvalidation` switch added to the three `_fg` samples; about an hour of machine time | A scheduled job on a dedicated machine; not justified for three samples and one developer |

**Recommendation.** Do follow-up 1 in Phase 1, before the A/B, because it is what makes the A/B a
proof. Bundle follow-up 2 into the same header bump only if 6.2 has already chosen A or D; otherwise
leave it. Run GBV by hand before each phase gate rather than standing up a machine for it, and record
the run in the phase's exit evidence. If the project gains a CI machine with a D3D12 GPU, move the GBV
run there and make it a gate; until then it is a checklist item.
