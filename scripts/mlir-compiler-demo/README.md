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
# Convert simple linear model through FX->MLIR
python simple_linear.py
# Convert partial LLM decode block through FX->MLIR
python simple_llm.py
```