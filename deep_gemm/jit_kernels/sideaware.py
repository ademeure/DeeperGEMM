import torch
import ctypes

_lib = None
_cpu_side_index = None
_gpu_side_index = None
_torch_side_index = None
_info = (0, 0, 0, 0, 0)
_info_str = {"num_sms": 0, "side0": 0, "side1": 0, "min": 0, "hash": 0}

# -----------------------------------------------------------------------------
# Externally visible API (+torch.ops.sideaware.[memcpy/one_to_one/elementwise])
# -----------------------------------------------------------------------------
def sideaware_enabled():
    return _lib and 1 or 0

# Create custom elementwise kernels (returns id for sideaware_elementwise)
def sideaware_compile(header_code: bytes, force_recompile: bool = True) -> int:
    return _lib.sideaware_compile(header_code, force_recompile)

# GPU SM side metadata (which SM is on which side, SMs per side, etc...)
def sideaware_torch_side_index():
    global _torch_side_index
    if _torch_side_index is None:
        _torch_side_index = torch.zeros(1, dtype=torch.uint8, device="cuda")
    return _torch_side_index # torch.uint8 tensor of size num_sms
def sideaware_gpu_side_index():
    return _gpu_side_index # gpu buffer of size num_sms
def sideaware_cpu_side_index():
    return _cpu_side_index # cpu buffer of size num_sms
def sideaware_info():
    return _info_str # {"num_sms", "side0", "side1", "min", "hash"}
def sideaware_info_raw():
    return _info # (num_sms, side0, side1, min, hash)


# Load sideaware.so library both directly and through CUDAPluggableAllocator
def sideaware_init(path = 'sideaware.so'):
    sideaware_alloc = torch.cuda.memory.CUDAPluggableAllocator(path, 'sideaware_malloc_auto', 'sideaware_free_auto')
    torch.cuda.memory.change_current_allocator(sideaware_alloc)

    global _lib
    _lib = ctypes.CDLL(path)

    # Define C-style function signatures
    _lib.sideaware_compile.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    _lib.sideaware_compile.restype = ctypes.c_int

    _lib.get_sm_side_summary.argtypes = []
    _lib.get_sm_side_summary.restype = ctypes.POINTER(ctypes.c_int * 5)

    _lib.fill_gpu_side_index.argtypes = [ctypes.c_void_p]
    _lib.fill_gpu_side_index.restype = None
    _lib.get_gpu_side_index.argtypes = []
    _lib.get_gpu_side_index.restype = ctypes.c_void_p
    _lib.get_cpu_side_index.argtypes = []
    _lib.get_cpu_side_index.restype = ctypes.c_void_p

    _lib.sideaware_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    _lib.sideaware_memcpy.restype = None
    _lib.sideaware_one_to_one.argtypes = [ctypes.c_int, ctypes.c_size_t,    # kernel_id, num_bytes
                                        ctypes.c_void_p, ctypes.c_void_p,   # out0, in0
                                        ctypes.c_int, ctypes.c_void_p]      # device, stream
    _lib.sideaware_one_to_one.restype = None
    _lib.sideaware_elementwise.argtypes = [ctypes.c_int, ctypes.c_size_t,   # kernel_id, num_bytes
                                        ctypes.c_void_p, ctypes.c_void_p,   # out0, out1
                                        ctypes.c_void_p, ctypes.c_void_p,   # out2, out3
                                        ctypes.c_void_p, ctypes.c_void_p,   # in0, in1
                                        ctypes.c_void_p, ctypes.c_void_p,   # in2, in3
                                        ctypes.c_void_p, ctypes.c_void_p,   # sideband_ptr, sideband_value
                                        ctypes.c_int, ctypes.c_int,         # parallel_chunks, forced_sm_per_side
                                        ctypes.c_int, ctypes.c_void_p]      # device, stream
    _lib.sideaware_elementwise.restype = None

    # Define PyTorch custom operations for memcpy/one_to_one/elementwise
    def direct_register_custom_op(op_lib, op_name, op_func, mutates_args):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
        op_lib.define(op_name + schema_str)
        op_lib.impl(op_name, op_func, "CUDA")

    sideaware_lib = torch.library.Library("sideaware", "FRAGMENT")
    direct_register_custom_op(sideaware_lib, "memcpy", sideaware_memcpy, mutates_args=(["dst"]))
    direct_register_custom_op(sideaware_lib, "one_to_one", sideaware_one_to_one, mutates_args=(["dst"]))
    direct_register_custom_op(sideaware_lib, "elementwise", sideaware_elementwise,
                              mutates_args=(["out0", "out1", "out2", "out3", "sideband_tensor"]))

    # Initialize sideaware metadata
    global _info, _info_str, _torch_side_index, _gpu_side_index, _cpu_side_index
    _info = tuple(_lib.get_sm_side_summary().contents)
    _info_str = { "num_sms": _info[0], "side0": _info[1], "side1": _info[2], "min": _info[3], "hash": _info[4] }

    _torch_side_index = torch.zeros(_info_str["num_sms"], dtype=torch.uint8, device="cuda")
    _lib.fill_gpu_side_index(_torch_side_index.data_ptr())
    _gpu_side_index = _lib.get_gpu_side_index()
    _cpu_side_index = _lib.get_cpu_side_index()

    # Print metadata (shows we are done with initialization)
    print(f"L2 Side Aware metadata: {_info_str}")

# -----------------------------------------------------------------------------
# Exposed via torch.ops.sideaware.[memcpy/one_to_one/elementwise]() only
# -----------------------------------------------------------------------------

# Sideaware memcpy (i.e. "default kernel" when no custom kernel is provided via sideaware_compile)
def sideaware_memcpy(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Validate inputs
    assert dst.device.type == "cuda" and src.device.type == "cuda", "Both tensors must be on CUDA"
    assert dst.dtype == src.dtype, "Source and destination must have the same dtype"
    assert dst.numel() >= src.numel(), "Destination tensor must be at least as large as source"

    # Get pointers and size
    dst_ptr = dst.data_ptr()
    src_ptr = src.data_ptr()
    num_bytes = src.numel() * src.element_size()

    # Make sure src and dst are contiguous and aligned
    assert src.is_contiguous(), "src must be contiguous"
    assert dst.is_contiguous(), "dst must be contiguous"
    assert (dst_ptr % 16 == 0) and (src_ptr % 16 == 0), "dst and src must be 16-byte aligned"

    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    _lib.sideaware_memcpy(dst_ptr, src_ptr, num_bytes, device, stream)

# Sideaware single-input / single-output elementwise API (simple version)
def sideaware_one_to_one(kernel_id: int, dst: torch.Tensor, src: torch.Tensor) -> None:
    # Validate inputs
    src_bytes = dst is not None and dst.numel() * dst.element_size() or 0
    dst_bytes = dst is not None and dst.numel() * dst.element_size() or 0
    num_bytes = max(src_bytes, dst_bytes)
    assert num_bytes > 0

    # Make sure src and dst are contiguous and aligned
    dst_ptr = dst is not None and dst.data_ptr() or 0
    src_ptr = src_bytes and src.data_ptr() or 0
    assert src is None or (src.is_contiguous() and src.device.type == "cuda"), "src must be contiguous"
    assert dst is None or (dst.is_contiguous() and dst.device.type == "cuda"), "dst must be contiguous"
    assert dst_ptr % 16 == 0 and src_ptr % 16 == 0, "dst and src must be 16-byte aligned"

    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    _lib.sideaware_one_to_one(kernel_id, num_bytes, dst_ptr, src_ptr, device, stream)

# Sideaware multi-input / multi-output elementwise API (advanced version)
def sideaware_elementwise(kernel_id: int,
                          out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor, out3: torch.Tensor,
                          in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, in3: torch.Tensor,
                          sideband_tensor: torch.Tensor = None, sideband_value: int = 0,
                          parallel_chunks: int = 0, forced_sm_per_side: int = 0) -> None:
    # Validate inputs
    src_bytes = out0 is not None and out0.numel() * out0.element_size() or 0
    dst_bytes = out0 is not None and out0.numel() * out0.element_size() or 0
    num_bytes = max(src_bytes, dst_bytes)
    assert num_bytes > 0

    # Make sure src and dst are contiguous and aligned
    out0_ptr = out0 is not None and out0.data_ptr() or 0
    out1_ptr = out1 is not None and out1.data_ptr() or 0
    out2_ptr = out2 is not None and out2.data_ptr() or 0
    out3_ptr = out3 is not None and out3.data_ptr() or 0
    in0_ptr = in0 is not None and in0.data_ptr() or 0
    in1_ptr = in1 is not None and in1.data_ptr() or 0
    in2_ptr = in2 is not None and in2.data_ptr() or 0
    in3_ptr = in3 is not None and in3.data_ptr() or 0
    sideband_ptr = sideband_tensor is not None and sideband_tensor.data_ptr() or 0

    assert in0 is None or (in0.is_contiguous() and in0.device.type == "cuda"), "in0 must be contiguous"
    assert in1 is None or (in1.is_contiguous() and in1.device.type == "cuda"), "in1 must be contiguous"
    assert in2 is None or (in2.is_contiguous() and in2.device.type == "cuda"), "in2 must be contiguous"
    assert in3 is None or (in3.is_contiguous() and in3.device.type == "cuda"), "in3 must be contiguous"
    assert out0 is None or (out0.is_contiguous() and out0.device.type == "cuda"), "out0 must be contiguous"
    assert out1 is None or (out1.is_contiguous() and out1.device.type == "cuda"), "out1 must be contiguous"
    assert out2 is None or (out2.is_contiguous() and out2.device.type == "cuda"), "out2 must be contiguous"
    assert out3 is None or (out3.is_contiguous() and out3.device.type == "cuda"), "out3 must be contiguous"
    assert out0_ptr % 16 == 0 and out1_ptr % 16 == 0 and out2_ptr % 16 == 0 and out3_ptr % 16 == 0, "16B alignment"
    assert in0_ptr % 16 == 0 and in1_ptr % 16 == 0 and in2_ptr % 16 == 0 and in3_ptr % 16 == 0, "16B alignment"

    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    _lib.sideaware_elementwise(kernel_id, num_bytes,
                              out0_ptr, out1_ptr, out2_ptr, out3_ptr,
                              in0_ptr, in1_ptr, in2_ptr, in3_ptr,
                              sideband_ptr, sideband_value, parallel_chunks, forced_sm_per_side, device, stream)
