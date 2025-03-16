import copy
import os
import torch
from typing import Any, Dict

from ..jit import build, cpp_format, generate, Runtime
from .utils import ceil_div

class JITTuner:
    def __init__(self) -> None:
        self.tuned = {}

    def compile_and_tune(self, name: str, keys: Dict[str, Any], space: tuple,
                         includes: tuple, arg_defs: tuple, template: str, args: tuple) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        keys = {k: keys[k] for k in sorted(keys.keys())}
        signature = (name, f'{keys}')
        if signature in self.tuned:
            if os.getenv('DG_JIT_DEBUG', None):
                print(f'Using cached JIT kernel {name} with keys {keys}')
            return self.tuned[signature]

        if os.getenv('DG_JIT_DEBUG', None):
            print(f'Auto-tuning JIT kernel {name} with keys {keys}')

        # TODO: dynamic/automatic tuning of unroll factor
        # TODO: manual unrolling by using template.py to copy the code multiple times?
        # TODO: handle tail better, because BLOCK_K=8192 means 31 iterations which is prime :(
        # TODO: rewrite all this to be helpful with DOUBLE_PUMP mode (and/or copy-paste instead of unroll)
        if not "NUM_UNROLL" in keys and "K" in keys:
            # Find largest divisor of loop iteration count that's no greater than max_unroll
            max_unroll = 15
            loop_iterations = max(1, ceil_div(keys["K"], 256) - 1) # 1 per 2 BLOCK_K
            num_unroll = min(loop_iterations, max_unroll)
            while loop_iterations % num_unroll != 0:
                num_unroll -= 1
            if (loop_iterations >= 16 and num_unroll <= 4):
                num_unroll = 8
            keys["NUM_UNROLL"] = num_unroll

        assert signature not in self.tuned
        assert args is not None
        space = (dict(), ) if len(space) == 0 else space

        kernels = []
        for tuned_keys in space:
            assert isinstance(tuned_keys, dict)
            full_keys = copy.deepcopy(keys)
            full_keys.update(tuned_keys)
            code = generate(includes, arg_defs, cpp_format(template, full_keys))

            # Illegal build must raise errors
            kernels.append((build(name, arg_defs, code), tuned_keys))

        best_runtime, best_time, best_keys = None, None, None
        for runtime, tuned_keys in kernels:
            if len(space) > 1:
                # Check kernel validity
                return_code = runtime(*args)
                if return_code != 0:
                    # Pass illegal kernels, e.g. insufficient shared memory capacity
                    if os.getenv('DG_JIT_DEBUG', None):
                        print(f'Illegal JIT kernel {name} with keys {keys} and tuned keys {tuned_keys}: error code {return_code}')
                    continue

                # Measure performance with L2 flush and a large GEMM kernel before to reduce overhead between kernels
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
                torch.randn((8192, 8192), dtype=torch.float, device='cuda') @ torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                start_event.record()
                for i in range(20):
                    assert runtime(*args) == 0
                end_event.record()
                end_event.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = 0

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if os.getenv('DG_JIT_DEBUG', None):
                print(f'Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}')
        assert best_runtime is not None, f'Failed to tune JIT kernel {name} with keys {keys}'

        # Cache the best runtime and return
        if os.getenv('DG_JIT_DEBUG', None) or os.getenv('DG_PRINT_AUTOTUNE', None):
            print(f'Best JIT kernel {name} with keys {keys} has tuned keys {best_keys} and time {best_time}')
        self.tuned[signature] = best_runtime
        return best_runtime


jit_tuner = JITTuner()
