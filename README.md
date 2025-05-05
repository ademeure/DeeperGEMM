# DeeperGEMM

New version, it's pretty cool (if you consider inline PTX cool).

L2 Side Optimization for B Matrix is now based on cuda-side-boost: https://github.com/ademeure/cuda-side-boost

Not planning to support this any further or provide in-depth documentation due to lack of time. There are also optimizations on the DeepGEMM repo that'd be hard to integrate - they've definitely closed the gap, but DeeperGEMM remains faster for many shapes.

If you have a specific question, feel free to get in touch though! (unless it's about the inline PTX to load the B matrix because it was provided to me directly by god, or maybe the devil, I'm not sure).

# Previous Release Notes

- L2 cache side awareness for B matrix (store tiles of B on the same side of the L2 as the SMs processing that part of the matrix).
- Improved TMA load pipelining that doesn’t rely on unrolling, so there are no bubbles between tiles, and reuse memory for D.
- Optimized tile output overlapped with 1st matmuls of next tile, and shared memory padding reducing bank conflicts by 8x.
- Improved GEMM pipelining with dual accumulators so every warpgroup has a GEMM running nearly all the time (instead of relying on inter-warpgroup parallelism).
- Optional support for 256x block size (instead of 128x) that halves the number of FMAs (probably okay for inference?) with highly optimized pipelining between GEMMs.
- Lots of small optimzations adding up to a lot of performance

UPDATE: I'm planning to release an updated & bugfixed version in a few days with a custom PyTorch memory allocator that will significantly reduce the overhead of "L2 side awareness" and provide a simple way to create L2 aware elementwise 1:1 kernels.

I'll also write explanations of the existing optimisations, feel free to let me know if you have any thoughts or things that don't make sense!

## DeeperGEMM

Deepseek’s DeepGEMM delivers great FP8 matrix multiplication performance with fine-grained scaling on NVIDIA Hopper GPUs using custom PTX, something that seemed nearly impossible previously given the lack of hardware support for micro-tensor scaling (added in Blackwell).

It’s very fast… so let’s make it *even way faster*!

## Architecture

See previous versions of README.md for a small amount of not very useful additional information.
