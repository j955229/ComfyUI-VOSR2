# Avoiding OOM in ComfyUI custom nodes

Notes written up after two real out-of-memory incidents in `VOSR2Upscale`
(2026-09-06), kept here because the underlying mistake generalizes well
beyond this node. Both incidents were the *same* bug — memory that should
have been bounded to "one batch item" instead scaled with the whole batch —
surfacing on two different memory pools in a row as each fix pushed the
problem to the next stage of the pipeline. See `run_vosr2()` in
`inference.py` for the code these notes describe.

## How ComfyUI actually hands you "a video"

There is no `VIDEO` tensor type in the input/output contract most custom
nodes see. An image is `IMAGE`: a `[B, H, W, C]` float tensor in `[0, 1]`.
A video, once a loader node (VideoHelperSuite's "Load Video", core video
nodes, etc.) has decoded it, is **the exact same `IMAGE` tensor** — every
frame stacked into the batch dimension `B`. (Newer ComfyUI versions do have
a `comfy_api` `VIDEO` type for passing an *encoded* video around without
decoding it to frames, used by nodes that only need to re-encode or mux —
but the moment something needs to run per-pixel model inference on the
content, it gets decoded to an `IMAGE` batch like any other.)

This matters because **nothing in the type system tells you which one you
have.** A node written and tested against "an image, maybe batched 1-4x for
a few variations" will, unmodified, also accept a 150-frame video as a
`B=150` `IMAGE` batch — and every assumption that was fine at `B=4` gets
tested at `B=150` the first time someone plugs in a video loader instead of
`LoadImage`. Batch size is not a hint about intent; treat any code that
scales with `B` as code that will eventually see video-sized batches, even
if the node was designed with stills in mind.

## Incident 1: tiling bounded per-item VRAM, not per-batch VRAM

`VOSR2Upscale`'s tiled path processes one image at a time specifically so a
single large image doesn't need its whole activation footprint resident at
once. The bug: the *batch* loop around that per-item tiling still did

```python
outputs_pm1 = torch.cat([
    _run_tiled_single(model, resized01[i:i + 1], ...)
    for i in range(b)
], dim=0)
```

Every iteration's full-resolution decoded output stayed GPU-resident in the
Python list — nothing freed or moved it — until every item was done and the
final `torch.cat` ran. Tiling capped the cost of *processing* one item; it
did nothing to cap the cost of *holding onto* every finished item. Peak VRAM
was `batch_size × output_resolution`, not `output_resolution` — for a
handful of images that's invisible; for a batch of video frames it's the
whole clip's decoded output sitting in VRAM simultaneously.

**Fix:** move each item off the GPU the moment it's done, before it joins
anything that outlives the loop:

```python
outputs_pm1 = torch.cat([
    _run_tiled_single(model, resized01[i:i + 1], ...).cpu()
    for i in range(b)
], dim=0)
```

**General lesson:** tiling/chunking a single item's *compute* and bounding
a *batch loop's* memory are two separate problems. Solving the first one
doesn't solve the second. If a loop produces one output per batch item,
ask separately: "what is still alive, and on what device, once this
iteration returns?"

## Incident 2: the same mistake, one stage later, on a different memory pool

Fixing incident 1 immediately surfaced a second one, in the very next line.
Color alignment ran on the *concatenated* batch, after the loop:

```python
decoded01 = (outputs_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
aligned01 = apply_color_alignment(decoded01, resized01, color_alignment)
```

`apply_color_alignment`'s wavelet/AdaIN modes run a Gaussian blur
(`F.conv2d`) over the *entire* tensor in one call. Once `outputs_pm1` was
CPU (from the incident-1 fix), this became a single **system RAM**
allocation sized to the whole batch's decoded output — for a video-sized
batch at a real upscale factor, tens of GB, on a machine that had already
committed VRAM's neighbor, ordinary RAM, to other loaded models.

The error looked completely different from incident 1 —
`torch.OutOfMemoryError` on `cuda:0` versus a generic
`RuntimeError: [enforce fail at alloc_cpu.cpp:117] ... DefaultCPUAllocator`
— because it was on a different device. **It was the same bug**: an
operation sized to the whole batch instead of one item, just moved from
VRAM to RAM by the previous fix rather than eliminated.

**Fix:** push color alignment *into* the per-item loop, so it runs on one
item's pixels at a time, right where that item is already isolated:

```python
aligned_items = []
for i in range(b):
    item_pm1 = _run_tiled_single(model, resized01[i:i + 1], ...).cpu()
    item01 = (item_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
    aligned_items.append(apply_color_alignment(item01, resized01[i:i + 1].cpu(), color_alignment))
aligned01 = torch.cat(aligned_items, dim=0)
```

**General lesson:** "I already fixed the memory problem" is only true for
the specific tensor and specific stage you changed. Any later stage that
re-touches the *whole batch* — postprocessing, color correction,
sharpening, resizing, saving, metrics — reintroduces exactly the same
`O(batch)` cost, just on whatever memory pool that stage happens to run on.
Grep your pipeline for anything downstream of a per-item loop that operates
on the re-concatenated result, not on that loop's individual items.

## Checklist for writing (or reviewing) a per-item/tiled node

- **Every stage independently, not just the "expensive" one.** Preprocess,
  encode, model forward, decode, postprocess: each one needs its own
  per-item memory bound. A single un-tiled stage anywhere in the chain — even
  a "cheap" one like a color-space conversion — reintroduces batch-size
  scaling for the whole pipeline, no matter how carefully the others are
  bounded.
- **Don't pre-materialize the whole batch at final resolution.** Resizing
  or otherwise transforming an entire batch to its *final*, largest size
  before any per-item tiling begins creates one big resident tensor for the
  full run. Do that transform per-item, just-in-time inside the loop, if
  the final size is much larger than the input size (as with an upscaler).
- **Move finished items off the accelerator immediately**, not at the end
  of the batch. `.cpu()` (or wherever they need to end up) belongs right
  where an item is produced, not deferred to a "finalize" step that runs
  after everything has accumulated.
- **Concatenate last, and as cheaply as possible.** By the time you
  `torch.cat` finished items back together, there should be nothing left to
  *compute* — only stacking already-final tensors. If a `cat`'d tensor is
  about to be fed into more processing, that processing should probably
  have happened before the `cat`, per-item.
- **Prefer ComfyUI's own memory/dtype machinery over hand-rolled device
  code** (`comfy.model_management`, `ModelPatcher`, `comfy.ops`,
  `comfy.utils.load_torch_file`) — it already understands offloading and
  lowvram patterns that hand-written `.to(device)` calls don't, and keeps
  a node's device handling consistent with how the rest of ComfyUI manages
  VRAM pressure across a whole running graph, not just one node.
- **Warn before the crash, not just after.** `run_vosr2()` logs an explicit
  warning when the target resolution exceeds a known-safe threshold with
  tiling disabled — a crash is a debugging session, a log line before it is
  a fix. Any node with a resolution- or batch-dependent memory cliff should
  say so before hitting it, not rely on the stack trace to explain itself.
- **Test at video-shaped batch sizes, not just image-shaped ones.** A batch
  of 1-4 is a different regime than a batch of 100+. If a node might ever
  see decoded video frames, test it at a batch size that actually
  represents a short clip, not just a handful of stills — that's the gap
  that let both incidents above ship unnoticed until someone tried it on a
  real video-length batch.
- **Read which memory pool an OOM names before assuming the fix is done.**
  `torch.OutOfMemoryError` on a `cuda:N` device means something is
  GPU-resident that shouldn't be; a CPU allocator failure
  (`alloc_cpu.cpp`, plain `RuntimeError`) means the same class of mistake
  has simply moved to system RAM — often *because* of a previous fix that
  moved data off the GPU without also bounding what happens to it next.
  Chasing an OOM one stage at a time, as happened here, is normal; the
  second error is progress, not a regression, if it's further down the
  pipeline than the first one was.
