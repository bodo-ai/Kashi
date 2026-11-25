"""Backend-neutral IREE GPU dispatch ABI and workgroup metadata."""

import ctypes
import operator
import queue
import re
import threading
import time
import weakref

import torch

_PENDING = queue.SimpleQueue()
_RETAIN_LOCK = threading.Lock()
_RETAIN_THREAD = None
_UNRELEASED = []
_CAPTURE_LOCK = threading.RLock()
_CAPTURE_HOOKS = None
_CAPTURES = {}
_CAPTURE_QUERY = None


def _hip_runtime_path():
    """Resolve PyTorch's loaded HIP library without mixing runtime contexts."""
    from kashi.iree_runtime import _mapped_hip_libraries

    paths = _mapped_hip_libraries()
    if len(paths) != 1:
        raise RuntimeError(f"Cannot uniquely identify PyTorch HIP library: {paths}")
    return next(iter(paths))


def _capture_identity(stream):
    """Return the driver's unique capture ID, not merely a reusable stream."""
    global _CAPTURE_QUERY
    with _CAPTURE_LOCK:
        if _CAPTURE_QUERY is None:
            if torch.version.hip and not torch.version.cuda:
                library, name = _hip_runtime_path(), "hipStreamGetCaptureInfo"
            elif torch.version.cuda and not torch.version.hip:
                library, name = "libcuda.so.1", "cuStreamGetCaptureInfo"
            else:
                raise RuntimeError(
                    "Cannot identify GPU graph capture on this PyTorch build"
                )
            try:
                lib = ctypes.CDLL(library)
                query = getattr(lib, name)
            except (OSError, AttributeError) as exc:
                raise RuntimeError(
                    "GPU graph capture identification is unavailable"
                ) from exc
            query.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            query.restype = ctypes.c_int
            # Keep the library loaded as long as the ctypes function is cached.
            _CAPTURE_QUERY = (lib, query)
        _, query = _CAPTURE_QUERY
    status, identity = ctypes.c_int(), ctypes.c_uint64()
    result = query(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.byref(status),
        ctypes.byref(identity),
    )
    if result or status.value != 1:
        raise RuntimeError("Could not identify active GPU graph capture")
    return identity.value


def _capture_started(graph):
    """Hooks only inspect capture state; they must never submit GPU work."""
    stream = torch.cuda.current_stream()
    key = (stream.device, stream.cuda_stream)
    with _CAPTURE_LOCK:
        # Clear stale mappings even if capture identity cannot be queried.
        _CAPTURES.pop(key, None)
        identity = _capture_identity(stream)
        _CAPTURES[key] = (weakref.ref(graph), identity)


def _capture_ended(graph):
    with _CAPTURE_LOCK:
        for key, (ref, _) in list(_CAPTURES.items()):
            if ref() is None or ref() is graph:
                del _CAPTURES[key]


def _ensure_capture_hooks():
    """Install lifecycle hooks during kernel creation, before capture begins."""
    global _CAPTURE_HOOKS
    with _CAPTURE_LOCK:
        if _CAPTURE_HOOKS is not None:
            return
        graphs = getattr(torch.cuda, "graphs", None)
        begin = getattr(graphs, "register_graph_capture_start_hook", None)
        end = getattr(graphs, "register_graph_capture_end_hook", None)
        retain = getattr(torch.cuda.CUDAGraph, "retain_object", None)
        if not (callable(begin) and callable(end) and callable(retain)):
            _CAPTURE_HOOKS = ()
            return
        first = begin(_capture_started)
        try:
            second = end(_capture_ended)
        except Exception:
            first.remove()
            raise
        _CAPTURE_HOOKS = (first, second)


def _retain_captured_bindings(stream, bindings):
    """Bind external owners to the exact graph through all future replays."""
    key = (stream.device, stream.cuda_stream)
    with _CAPTURE_LOCK:
        entry = _CAPTURES.get(key)
        graph = entry[0]() if entry is not None else None
        if graph is None or entry[1] != _capture_identity(stream):
            raise RuntimeError("Cannot identify active GPU graph capture safely")
        retain = getattr(graph, "retain_object", None)
        if not callable(retain):
            raise RuntimeError("GPU graph capture cannot retain tensor owners")  # noqa: TRY004
        retain(tuple(bindings), synchronize_before_release=True)


def _release_completed_bindings():
    """Keep external tensor owners alive until the recorded work completes."""
    while True:
        event, bindings = _PENDING.get()
        try:
            # Host synchronization during another stream's global capture can
            # invalidate it. Querying an ordinary event is nonblocking.
            while not event.query():
                time.sleep(0.01)
        except Exception:  # noqa: BLE001 - any event failure leaves ownership uncertain
            # An error does not prove that the GPU has stopped using memory.
            # Keep owners indefinitely rather than risk freeing live storage.
            _UNRELEASED.append(bindings)
        finally:
            del bindings, event


def _launch_retaining_bindings(stream, bindings, launch):
    """Retain tensor owners, including DLPack deleters, until stream completion."""
    global _RETAIN_THREAD
    event = torch.cuda.Event()
    with _RETAIN_LOCK:
        if _RETAIN_THREAD is None:
            worker = threading.Thread(target=_release_completed_bindings, daemon=True)
            worker.start()
            _RETAIN_THREAD = worker
    # Allocate the tuple before launching: no exceptional path after launch may
    # let externally owned storage die while queued GPU work still references it.
    owners = tuple(bindings)
    try:
        launch()
    finally:
        try:
            event.record(stream)
            _PENDING.put((event, owners))
        except BaseException:
            _UNRELEASED.append(owners)
            raise


_SSA = r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?"
# IREE lowers signed and unsigned Torch integers to signless tensor elements;
# the generated instructions still preserve the source operation's semantics.
_FLOW_DTYPES = {
    "f64": (torch.float64,),
    "f32": (torch.float32,),
    "f16": (torch.float16,),
    "bf16": (torch.bfloat16,),
    "i1": (torch.bool,),
    "i8": (torch.int8, torch.uint8),
    "ui8": (torch.uint8,),
    "i16": (torch.int16, torch.uint16),
    "ui16": (torch.uint16,),
    "i32": (torch.int32, torch.uint32),
    "ui32": (torch.uint32,),
    "i64": (torch.int64, torch.uint64),
    "ui64": (torch.uint64,),
}


def _unsupported(export, what):
    """Report an ABI construct together with its executable export."""
    raise NotImplementedError(f"IREE GPU export {export}: unsupported {what}")


def _balanced(text, start, opening, closing):
    """Return the contents and end position of a balanced textual IR group."""
    depth = 0
    quoted = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if quoted:
            if char == '"' and not escaped:
                quoted = False
            escaped = char == "\\" and not escaped
            continue
        if char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if not depth:
                return text[start + 1 : pos], pos + 1
    raise ValueError(f"Unterminated {opening} group in IREE metadata")


def _split(text):
    """Split comma-separated MLIR values without splitting nested types."""
    parts, start, stack = [], 0, []
    pairs = {">": "<", ")": "(", "]": "[", "}": "{"}
    for i, char in enumerate(text):
        if char in "<([{":
            stack.append(char)
        elif char in ">)]}" and stack and stack[-1] == pairs[char]:
            stack.pop()
        elif char == "," and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def _phase(directory, suffix):
    """Find the unique requested compilation phase, never guessing a match."""
    paths = list((directory / "phases").glob(f"*.{suffix}.mlir"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one IREE {suffix} phase in {directory}, found {len(paths)}"
        )
    return paths[0].read_text()


def _executable(text, kind, name):
    """Extract the named executable, including its nested bodies."""
    pattern = rf"\b{re.escape(kind)}\.executable\s+(?:private|public)\s+@{re.escape(name)}(?=\s|\{{)"
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Missing {kind}.executable @{name}")
    start = text.find("{", match.end())
    if start < 0:
        raise RuntimeError(f"Missing body for {kind}.executable @{name}")
    return _balanced(text, start, "{", "}")[0]


def _function(text, export):
    """Extract an export's function arguments and body from flow or stream IR."""
    match = re.search(
        rf"\bfunc\.func\s+(?:(?:private|public)\s+)?@{re.escape(export)}\s*\(", text
    )
    if not match:
        raise RuntimeError(f"Missing function for IREE GPU export {export}")
    args, end = _balanced(text, match.end() - 1, "(", ")")
    body_start = text.find("{", end)
    if body_start < 0:
        raise RuntimeError(f"Missing function body for IREE GPU export {export}")
    body, _ = _balanced(text, body_start, "{", "}")
    return _split(args), body


def _offset_constants(body, binding_count, constant_count, export):
    """Identify stream allocation offsets that become zero for separate buffers.

    Stream may coalesce independent results in one allocation and pass their
    byte offsets as extra push constants. Our separate tensor allocations use
    offset zero. Trace SSA definitions, including multi-result util.assume.int,
    and reject an offset which also contributes to kernel computation.
    """
    definitions = {}
    expressions = {}
    operations = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "util.assume.int" in line and ":" not in line.split("util.assume.int", 1)[1]:
            i += 1
            while i < len(lines):
                line += " " + lines[i].strip()
                if re.search(r"\s:\s", lines[i]):
                    break
                i += 1
        i += 1
        match = re.match(rf"({_SSA})(?::(\d+))?\s*=\s*(.*)", line)
        if not match:
            if line.startswith("^bb"):
                for block_arg in re.findall(rf"({_SSA})\s*:", line):
                    definitions[block_arg] = []
            if line and not line.startswith(("return", "^", "}")):
                operations.append(("", line))
            continue
        name, count, rhs = match.groups()
        op = rhs.split(None, 1)[0]
        if count:
            if op != "util.assume.int":
                _unsupported(export, f"multi-result stream operation {op}")
            refs = re.findall(rf"({_SSA})\s*<[^>]*>", rhs)
            if len(refs) != int(count):
                _unsupported(export, "util.assume.int operands")
            for index, ref in enumerate(refs):
                definitions[f"{name}#{index}"] = [ref]
                expressions[f"{name}#{index}"] = ref
        elif op == "stream.binding.subspan":
            # The resulting dispatch tensor represents the whole binding, not
            # a use of the allocator offset by the payload computation.
            binding = re.match(r"stream\.binding\.subspan\s+(%arg\d+)\[", rhs)
            if not binding:
                _unsupported(export, f"stream binding subspan {rhs}")
            definitions[name] = [binding.group(1)]
        else:
            definitions[name] = re.findall(_SSA, rhs)
        if not count:
            expressions[name] = rhs
        operations.append((op, rhs))

    def deps(name, seen=None):
        """Trace an SSA value to stream function argument ordinals."""
        seen = set() if seen is None else seen
        if name in seen:
            return set()
        seen.add(name)
        match = re.fullmatch(r"%arg(\d+)", name)
        if match:
            return {int(match.group(1))}
        if name not in definitions:
            _unsupported(export, f"stream SSA value {name}")
        return set().union(*(deps(ref, seen) for ref in definitions[name]))

    def zero_value(name):
        """Evaluate an offset reconstruction with all push constants zero."""
        if re.fullmatch(r"%arg\d+", name):
            return 0
        rhs = expressions.get(name)
        if rhs is None:
            _unsupported(export, f"stream offset value {name}")
        if re.fullmatch(_SSA, rhs):
            return zero_value(rhs)
        constant = re.match(r"arith\.constant\s+(-?\d+)\s*:", rhs)
        if constant:
            return int(constant.group(1))
        unary = re.match(
            rf"arith\.(?:extui|extsi|index_castui|index_cast|trunci)\s+({_SSA})\s*:",
            rhs,
        )
        if unary:
            return zero_value(unary.group(1))
        binary = re.match(
            rf"arith\.(shli|ori|addi|subi|muli)\s+({_SSA})\s*,\s*({_SSA})\s*:", rhs
        )
        if binary:
            op, lhs, rhs = binary.groups()
            a, b = zero_value(lhs), zero_value(rhs)
            return {
                "shli": lambda: a << b,
                "ori": lambda: a | b,
                "addi": lambda: a + b,
                "subi": lambda: a - b,
                "muli": lambda: a * b,
            }[op]()
        _unsupported(export, f"stream offset expression {rhs}")

    offsets = set()
    for match in re.finditer(
        rf"stream\.binding\.subspan\s+(%arg\d+)\[({_SSA})\]", body
    ):
        binding = int(match.group(1)[4:])
        if binding >= binding_count:
            _unsupported(export, f"binding ordinal {binding}")
        if zero_value(match.group(2)) != 0:
            _unsupported(export, "nonzero fixed stream binding offset")
        offsets.update(deps(match.group(2)))
    # Offset-only constants cannot also affect an actual computation.  Ignore
    # the scalar reconstruction operations and binding metadata themselves.
    scalar_ops = {
        "arith.constant",
        "arith.extui",
        "arith.extsi",
        "arith.shli",
        "arith.ori",
        "arith.index_cast",
        "arith.index_castui",
        "arith.trunci",
        "util.assume.int",
        "stream.binding.subspan",
    }
    used = set()
    for op, rhs in operations:
        if op in scalar_ops:
            continue
        for ref in re.findall(_SSA, rhs):
            # Operation bodies contain region block arguments not defined here;
            # only follow values belonging to the function-level SSA graph.
            if ref in definitions or re.fullmatch(r"%arg\d+", ref):
                used.update(deps(ref))
    if offsets & used:
        _unsupported(export, "stream offset also used in computation")
    if any(n < binding_count or n >= binding_count + constant_count for n in offsets):
        _unsupported(export, "non-constant stream binding offset")
    return {n - binding_count for n in offsets}


def _scalar_type(arg, export):
    """Return a flow scalar type, rejecting representations not in the ABI."""
    match = re.match(r"%[^:]+:\s*(index|i32|i64|f32)\s*$", arg)
    if not match:
        _unsupported(export, f"flow argument {arg}")
    return match.group(1)


def _validate_tensor(tensor, specification, export):
    """Validate the dense tensor layout and collect declared dynamic sizes."""
    if tensor.is_neg() or tensor.is_conj():
        _unsupported(export, "lazy negative or conjugate tensor binding")
    match = re.search(
        r"!iree_tensor_ext\.dispatch\.tensor<(readonly|writeonly|readwrite):tensor<([^<>]+)>>",
        specification,
    )
    if not match:
        _unsupported(export, f"flow tensor type {specification}")
    parts = match.group(2).split("x")
    element = parts.pop()
    if element not in _FLOW_DTYPES:
        _unsupported(export, f"flow tensor element type {element}")
    if tensor.dtype not in _FLOW_DTYPES[element] or len(tensor.shape) != len(parts):
        raise ValueError(
            f"IREE GPU export {export}: tensor does not match {specification}"
        )
    dynamic = []
    for dim, actual in zip(parts, tensor.shape):
        if dim == "?":
            dynamic.append(actual)
        elif not dim.isdigit():
            _unsupported(export, f"flow tensor dimension {dim}")
        elif int(dim) != actual:
            raise ValueError(
                f"IREE GPU export {export}: tensor does not match {specification}"
            )
    return dynamic


def _affine_expr(expression, values, export):
    """Evaluate the integer affine expression grammar used for launch grids."""
    tokens = re.findall(r"ceildiv|floordiv|mod|[ds]\d+|\d+|[()+*\-]", expression)
    if "".join(tokens) != re.sub(r"\s+", "", expression):
        _unsupported(export, f"affine expression {expression}")
    pos = 0

    def parse(min_precedence=0):
        """Parse one affine subexpression with MLIR operator precedence."""
        nonlocal pos
        if pos >= len(tokens):
            _unsupported(export, f"affine expression {expression}")
        token = tokens[pos]
        pos += 1
        if token == "-":
            left = -parse(3)
        elif token == "(":
            left = parse()
            if pos >= len(tokens) or tokens[pos] != ")":
                _unsupported(export, f"affine expression {expression}")
            pos += 1
        elif token.isdigit():
            left = int(token)
        elif token in values:
            left = values[token]
        else:
            _unsupported(export, f"affine value {token}")
        precedence = {"+": 1, "-": 1, "*": 2, "floordiv": 2, "ceildiv": 2, "mod": 2}
        while pos < len(tokens) and precedence.get(tokens[pos], -1) >= min_precedence:
            op = tokens[pos]
            pos += 1
            right = parse(precedence[op] + 1)
            if op == "+":
                left += right
            elif op == "-":
                left -= right
            elif op == "*":
                left *= right
            elif right <= 0:
                _unsupported(export, f"affine divisor {right}")
            elif op == "ceildiv":
                left = -(-left // right)
            elif op == "floordiv":
                left //= right
            else:
                left %= right
        return left

    result = parse()
    if pos != len(tokens):
        _unsupported(export, f"affine expression {expression}")
    return result


def _grid(count_args, body, maps, workload, export):
    """Interpret the executable's host-side workgroup-count region."""
    if len(count_args) != len(workload) + 1 or "!hal.device" not in count_args[0]:
        _unsupported(export, "workgroup count arguments")
    values = {}
    for arg, value in zip(count_args[1:], workload):
        match = re.fullmatch(r"(%[^:]+):\s*index", arg)
        if not match:
            _unsupported(export, f"workgroup argument {arg}")
        values[match.group(1)] = operator.index(value)
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        ret = re.fullmatch(
            r"hal\.return\s+(.+?)\s*:\s*index\s*,\s*index\s*,\s*index", line
        )
        if ret:
            try:
                grid = tuple(values[name] for name in _split(ret.group(1)))
            except KeyError as exc:
                _unsupported(export, f"workgroup value {exc.args[0]}")
            if len(grid) != 3 or any(v < 0 or v > 0xFFFFFFFF for v in grid):
                _unsupported(export, f"workgroup grid {grid}")
            return grid
        match = re.fullmatch(
            rf"({_SSA})\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*index", line
        )
        if match:
            values[match.group(1)] = int(match.group(2))
            continue
        match = re.fullmatch(
            rf"({_SSA})\s*=\s*arith\.(addi|subi|muli|divui|ceildivui|minui|maxui)\s+({_SSA})\s*,\s*({_SSA})\s*:\s*index",
            line,
        )
        if match:
            dest, op, lhs, rhs = match.groups()
            a, b = values[lhs], values[rhs]
            if op in ("divui", "ceildivui") and b <= 0:
                _unsupported(export, f"workgroup divisor {b}")
            # Evaluate only the selected operation: unrelated division may have
            # a zero divisor even when the requested operation is valid.
            if op == "addi":
                values[dest] = a + b
            elif op == "subi":
                values[dest] = a - b
            elif op == "muli":
                values[dest] = a * b
            elif op == "divui":
                values[dest] = a // b
            elif op == "ceildivui":
                values[dest] = -(-a // b)
            elif op == "minui":
                values[dest] = min(a, b)
            else:
                values[dest] = max(a, b)
            continue
        match = re.fullmatch(
            rf"({_SSA})\s*=\s*affine\.(apply|min|max)\s+(#[\w.$-]+)\(([^)]*)\)(?:\[([^]]*)\])?",
            line,
        )
        if match:
            dest, op, map_name, dims, symbols = match.groups()
            if map_name not in maps:
                _unsupported(export, f"affine map {map_name}")
            dim_names, symbol_names, expressions = maps[map_name]
            actual_dims, actual_symbols = _split(dims), _split(symbols or "")
            if len(dim_names) != len(actual_dims) or len(symbol_names) != len(
                actual_symbols
            ):
                _unsupported(export, f"affine map operands {map_name}")
            env = {
                name: values[value]
                for name, value in zip(
                    dim_names + symbol_names, actual_dims + actual_symbols
                )
            }
            results = [_affine_expr(expr, env, export) for expr in expressions]
            if not results or (op == "apply" and len(results) != 1):
                _unsupported(export, f"affine {op} map {map_name}")
            values[dest] = (
                min(results)
                if op == "min"
                else max(results)
                if op == "max"
                else results[0]
            )
            continue
        _unsupported(export, f"workgroup count operation {line}")
    _unsupported(export, "workgroup count without hal.return")


def _metadata(directory, executable, export):
    """Read and validate the flow, stream, and HAL launch metadata."""
    flow = _executable(_phase(directory, "flow"), "flow", executable)
    stream = _executable(_phase(directory, "stream"), "stream", executable)
    targets = _phase(directory, "executable-targets")
    target = _executable(targets, "hal", executable)
    flow_args, _ = _function(flow, export)
    stream_args, stream_body = _function(stream, export)
    binding_count = sum("!stream.binding" in arg for arg in stream_args)
    if any("!stream.binding" not in arg for arg in stream_args[:binding_count]) or any(
        not re.fullmatch(r"%[^:]+:\s*i32", arg) for arg in stream_args[binding_count:]
    ):
        _unsupported(export, "stream function parameter order or constant type")
    constant_count = len(stream_args) - binding_count
    offsets = _offset_constants(stream_body, binding_count, constant_count, export)

    match = re.search(
        rf"\bhal\.executable\.export\s+(?:public|private)\s+@{re.escape(export)}\b.*?\bcount\s*\(",
        target,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"Missing HAL export/count for IREE GPU export {export}")
    count_args, end = _balanced(target, match.end() - 1, "(", ")")
    start = target.find("{", end)
    count_body, end = _balanced(target, start, "{", "}")
    attr_match = re.match(r"\s*<([^>]+)>", target[end:])
    if not attr_match:
        _unsupported(export, "missing workgroup launch attributes")
    attrs = attr_match.group(1)
    size_match = re.search(r"workgroup_size\s*=\s*\[([^]]+)\]", attrs)
    if not size_match:
        _unsupported(export, "missing workgroup_size")
    sizes = _split(size_match.group(1))
    if len(sizes) != 3 or any(not re.fullmatch(r"\d+\s*:\s*index", x) for x in sizes):
        _unsupported(export, f"workgroup_size {size_match.group(1)}")
    block = tuple(int(x.split(":")[0]) for x in sizes)
    shared_match = re.search(r"workgroup_local_memory\s*=\s*(\d+)", attrs)
    shared = int(shared_match.group(1)) if shared_match else 0
    maps = {}
    for m in re.finditer(
        r"^(#[\w.$-]+)\s*=\s*affine_map<\(([^)]*)\)(?:\[([^]]*)\])?\s*->\s*\(([^>]*)\)>",
        targets,
        re.MULTILINE,
    ):
        maps[m.group(1)] = (
            _split(m.group(2)),
            _split(m.group(3) or ""),
            _split(m.group(4)),
        )
    return (
        flow_args,
        binding_count,
        constant_count,
        offsets,
        _split(count_args),
        count_body,
        maps,
        block,
        shared,
    )


def _pack_scalar(value, kind, export):
    """Pack flow scalar arguments into IREE's little-endian 32-bit constants."""
    if kind == "f32":
        return [ctypes.c_uint32.from_buffer_copy(ctypes.c_float(value)).value]
    try:
        number = operator.index(value)
    except TypeError:
        _unsupported(export, f"{kind} scalar {type(value).__name__}")
    bits = 64 if kind in ("index", "i64") else 32
    if number < -(1 << (bits - 1)) or number >= (1 << bits):
        _unsupported(export, f"{kind} scalar outside {bits}-bit range")
    number &= (1 << bits) - 1
    return [number & 0xFFFFFFFF, number >> 32] if bits == 64 else [number]


def create_gpu_kernel(
    executable,
    export,
    metadata,
    aligns,
    launch,
    *,
    vendor,
    validate_device=None,
):
    """Construct the shared FX callable; launch is the vendor driver callback."""
    _ensure_capture_hooks()
    (
        flow_args,
        binding_count,
        constant_count,
        offsets,
        count_args,
        count_body,
        maps,
        block,
        shared,
    ) = metadata

    def kernel(
        *args, workload, argument_dims, result_shapes, result_dtypes, tied_operands
    ):
        """Allocate logical outputs and launch directly on the caller's stream."""
        # Dynamic shape operands are metadata, not extra kernel constants.
        if len(result_shapes) != len(result_dtypes) or len(result_shapes) != len(
            tied_operands
        ):
            raise ValueError(
                f"IREE {vendor.upper()} export {export}: inconsistent result metadata"
            )
        if len(flow_args) < len(args):
            _unsupported(export, "flow argument count")
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if not tensors or not tensors[0].is_cuda:
            _unsupported(export, f"missing {vendor.upper()} tensor device")
        device = tensors[0].device
        if any(not tensor.is_cuda or tensor.device != device for tensor in tensors):
            _unsupported(export, "mixed input devices")
        if any(not tensor.is_contiguous() for tensor in tensors):
            _unsupported(export, "non-contiguous tensor binding")
        inputs, scalars, dynamic_dims = [], [], []
        for arg, specification in zip(args, flow_args):
            if "!iree_tensor_ext.dispatch.tensor<" in specification:
                if not isinstance(arg, torch.Tensor):
                    _unsupported(export, f"non-tensor binding {specification}")
                dynamic_dims.extend(_validate_tensor(arg, specification, export))
                inputs.append(arg)
            else:
                scalars.extend(
                    _pack_scalar(arg, _scalar_type(specification, export), export)
                )
        if [operator.index(dim) for dim in argument_dims] != dynamic_dims:
            raise ValueError(
                f"IREE {vendor.upper()} export {export}: argument dimensions do not match tensors"
            )
        outputs, new_outputs = [], []
        with torch.cuda.device(device):
            if validate_device is not None:
                validate_device(device)
            for shape, dtype, tied in zip(result_shapes, result_dtypes, tied_operands):
                shape = tuple(operator.index(dim) for dim in shape)
                if any(dim < 0 for dim in shape):
                    raise ValueError(
                        f"IREE {vendor.upper()} export {export}: negative output shape {shape}"
                    )
                if tied == -1:
                    out = torch.empty(shape, dtype=dtype, device=device)
                    new_outputs.append(out)
                else:
                    if (
                        not isinstance(tied, int)
                        or tied < 0
                        or tied >= len(args)
                        or not isinstance(args[tied], torch.Tensor)
                    ):
                        _unsupported(export, f"tied operand {tied}")
                    out = args[tied]
                    if out.dtype != dtype or tuple(out.shape) != shape:
                        _unsupported(
                            export, f"tied operand {tied} result shape or dtype"
                        )
                outputs.append(out)
            output_specs = flow_args[len(args) :]
            if len(output_specs) != len(new_outputs) or any(
                "!iree_tensor_ext.dispatch.tensor<" not in spec for spec in output_specs
            ):
                _unsupported(export, "flow result binding signature")
            for output, specification in zip(new_outputs, output_specs):
                _validate_tensor(output, specification, export)
            bindings = inputs + new_outputs
            if len(bindings) != binding_count:
                _unsupported(
                    export, f"binding count {len(bindings)} != {binding_count}"
                )
            constants = []
            scalar_iter = iter(scalars)
            for index in range(constant_count):
                if index in offsets:
                    constants.append(0)
                else:
                    try:
                        constants.append(next(scalar_iter))
                    except StopIteration:
                        _unsupported(export, "missing flow scalar constants")
            if next(scalar_iter, None) is not None:
                _unsupported(export, "unmapped flow scalar constants")
            grid = _grid(count_args, count_body, maps, workload, export)
            if all(grid):
                for tensor, alignment in zip(bindings, aligns):
                    if not tensor.data_ptr():
                        _unsupported(export, "null tensor binding in nonempty launch")
                    if tensor.data_ptr() % alignment:
                        _unsupported(export, f"binding alignment {alignment}")
                stream = torch.cuda.current_stream(device)
                # PyTorch's stream allocator must keep every input and output
                # alive until this externally issued kernel has completed.
                for tensor in bindings:
                    tensor.record_stream(stream)
                if torch.cuda.is_current_stream_capturing():
                    _retain_captured_bindings(stream, bindings)
                    launch(device, stream, bindings, constants, grid, block, shared)
                else:
                    _launch_retaining_bindings(
                        stream,
                        bindings,
                        lambda: launch(
                            device, stream, bindings, constants, grid, block, shared
                        ),
                    )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    kernel.__name__ = re.sub(r"\W", "_", f"iree_{vendor}_{executable}_{export}")
    kernel.__module__ = __name__
    return kernel
