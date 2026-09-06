# OSWorld-G evaluation for Holo, on a machine with GPUs.
#
# The vLLM server and the benchmark runner live in the same container: the
# runner starts `vllm serve` itself (holo_osworld_g.py --model_path ...), so
# nothing has to be orchestrated from outside.
#
#   docker build -t osworld-g-holo .
#   docker run --rm --gpus all --ipc=host --shm-size=16g \
#       -v /models/Holo-3.1-pruned:/models/holo:ro \
#       -v "$PWD/results":/results \
#       --user "$(id -u):$(id -g)" \
#       osworld-g-holo --model_path /models/holo --model_name holo-pruned \
#                      --use_cache --output_path /results/holo_pruned_preds.json
#
# The benchmark (annotations + 264 screenshots, 124 MB) is baked in, so the
# image is self-contained. Weights are never baked -- mount them at /models.
ARG VLLM_TAG=v0.28.0
FROM vllm/vllm-openai:${VLLM_TAG}

# The base image ships torch, vllm, transformers, pillow, numpy, tqdm and
# requests. These are what the evaluation adds on top:
#   openai    -- client for the vLLM OpenAI-compatible endpoint
#   loguru    -- logging, used across the evaluation scripts
#   datasets  -- imported at the top of eval.py for its unused load_from_disk
#                path; GroundingEval._eval itself needs nothing
#
# Deliberately NOT installed from evaluation/requirements.txt: that file pins
# `concurrent-futures`, a name that does not exist on PyPI (concurrent.futures
# has been stdlib since 3.2), so installing it fails outright. `lmms-eval` and
# `qwen-agent` are left out too -- only the Jedi/Aguvis/Operator scripts import
# them, and they would drag their own torch pins into a working CUDA stack.
# `python3 -m pip` rather than `pip`: it pins the install to the same
# interpreter the entrypoint runs, so there is no way for the two to diverge.
# The base is Ubuntu, which ships python3 and no `python` symlink at all.
RUN python3 -m pip install --no-cache-dir \
        openai>=1.0.0 \
        loguru>=0.7.0 \
        datasets>=2.14.0

# Fail the build here rather than on the remote machine if the installs above
# disturbed the base image's CUDA stack.
RUN python3 -c "import vllm, torch, transformers, openai, loguru, datasets; \
print('vllm', vllm.__version__, '| torch', torch.__version__, \
'| transformers', transformers.__version__)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    VLLM_CACHE_ROOT=/cache/vllm \
    HOME=/cache \
    CACHE_DIR=/results

# World-writable so the container runs under any `--user`, which is what keeps
# bind-mounted results owned by the host account instead of by root.
RUN mkdir -p /cache/huggingface /cache/triton /cache/vllm /models /results \
    && chmod -R 777 /cache /results

WORKDIR /app
COPY benchmark/ ./benchmark/
COPY evaluation/ ./evaluation/

# The runner resolves ../benchmark/... relative to the evaluation directory,
# so keeping the repo layout is what makes the defaults work with no flags.
WORKDIR /app/evaluation
RUN python3 -c "import holo_osworld_g; print('runner imports clean')"

# The base image's entrypoint is `vllm serve`; this replaces it with the
# benchmark runner, which launches vLLM on its own.
ENTRYPOINT ["python3", "holo_osworld_g.py"]
CMD ["--help"]
