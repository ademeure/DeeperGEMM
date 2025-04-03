UPDATE: I'm planning to release an updated & bugfixed version in a few days with a custom PyTorch memory allocator that will significantly reduce the overhead of "L2 side awareness" and provide a simple way to create L2 aware elementwise 1:1 kernels.

I'll also write explanations of the existing optimisations, feel free to let me know if you have any thoughts or things that don't make sense!

# DeeperGEMM

Deepseek’s DeepGEMM delivers great FP8 matrix multiplication performance with fine-grained scaling on NVIDIA Hopper GPUs using custom PTX, something that seemed nearly impossible previously given the lack of hardware support for micro-tensor scaling (added in Blackwell).

It’s very fast… so let’s make it *even way faster*!

(((Highlights of best performance numbers + link to full results)))

(TODO: comparison with cuBLAS without micro-tensor scaling)

Key Features
- L2 cache side awareness for B matrix (store tiles of B on the same side of the L2 as the SMs processing that part of the matrix).
- Improved TMA load pipelining that doesn’t rely on unrolling, so there are no bubbles between tiles, and reuse memory for D.
- Optimized tile output overlapped with 1st matmuls of next tile, and shared memory padding reducing bank conflicts by 8x.
- Improved GEMM pipelining with dual accumulators so every warpgroup has a GEMM running nearly all the time (instead of relying on inter-warpgroup parallelism).
- Optional support for 256x block size (instead of 128x) that halves the number of FMAs (probably okay for inference?) with highly optimized pipelining between GEMMs.
- Lots of small optimzations adding up to a lot of performance

Future Plans
- Tidying, Bug Fixing & Explanations
- Support for “Metadata-free Drop-in FP8” by embedding the scaling factors inside the LSB bits of the FP8 data itself (encoded as parity) that should be easier to integrate into existing codebases.
- Use TMA tensor instructions instead of multiple 1D TMAs to massively reduce cost of TMAs for “L2 side aware” B matrix (currently a big bottleneck).
- Maybe add back experimental support for “Hybrid Cluster Sizes”: only up to 120 SMs can be used with a cluster size of 8, so run 15x8+6x2! (worked before L2 side awareness changes that makes it harder, it improved power but reduced peak performance).

DeepGEMM’s architecture: Benefits & Limitations

1. Unrolled pipeline stages

The original code relies on the number of pipeline stages (i.e. input A/B blocks loaded in advance) being known at compile time to be able to fully unroll all the key loops that depend on it.

This means there are no dynamic address calculations for the shared memory of input data, barriers, etc… it’s nearly all known at compile time and optimized away, e.g.:

[example from original code]

This is great most of the time, but there’s a terrible price to pay: if the number of blocks (i.e. K/128) isn’t a multiple of the number of pipeline stages (practically never the case), the remaining pipeline stages are wasted as the next tile starts back at pipeline stage 0.

This massively reduces latency tolerance in the worst case with 1 more block than stages… which is *exactly* the case for GPT2-124M where K=768 (=> 768/128 ‎ = 6 blocks per tile, but there’s only space for 5 pipeline stages when BLOCK_N‎ = 128) making it completely unsuitable for nanogpt modded speedruns! (clearly the only workload that matters! Nobody needs more than 640K parameters… err, sorry, 124M).

And in the opposite scenario where K is very large, the shared memory space for the final tile output D is always reserved, even though it is only needed for a very short time at the end of each tile; in most cases, size(D) < size(AB), so a stage’s memory could be reused for D instead, increasing maximum blocks from 5 to 6 in certain cases (which corresponds to ~1280 cycles of memory latency tolerance instead of 1024 depending on how you calculate it).

But making this dynamic is going to be very expensive… or is it?

(... TODO ...)
