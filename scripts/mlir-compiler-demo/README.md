### Running scripts:

``` shell
# Build vLLM from source
git clone https://github.com/bodo-ai/vllm.git
uv venv
source .venv/bin/activate
cd vllm

# build from source (CUDA enabled environment):
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto

# build from source (CPU)
uv pip install -r requirements/cpu.txt
uv pip install -e .
cd ..

# install torch-mlir + tabulate
uv pip install -r requirements.txt
# Roundtrip simple linear model through FX->MLIR->FX
python simple_linear.py
# Roundtrip simple llm model (NOTE: currently will throw an error):
# [rank0]:     raise NotImplementedError(
# [rank0]:     ...<2 lines>...
# [rank0]:     )
# [rank0]: torch._dynamo.exc.BackendCompilerFailed: backend='mlir_roundtrip_backend' raised:
# [rank0]: NotImplementedError: Could not deduce type from value info: tensor_meta=None, val=<torch._library.fake_class_registry.FakeScriptObject object at 0x33d2c5550> <class 'torch._library.fake_class_registry.FakeScriptObject'>
python simple_llm.py
```