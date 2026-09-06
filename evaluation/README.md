# Evaluation

We provide evaluation scripts for Jedi on Screenspot-v2, Screenspot-pro, and OSWorld-G. You can go into `evaluation` folder and run the scripts.

## Jedi on Screenspot-v2
Fetch the benchmark dataset from [here](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2/tree/main).
Run the following command to evaluate on Screenspot-v2:
```bash
python qwen25_vllm_screenspot_v2.py --annotation_path <path_to_annotation(json file)> --model_path <path_to_model> --image_dir <path_to_image_dir>
```

## Jedi on Screenspot-pro
Fetch the benchmark dataset from [here](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro).
Run the following command to evaluate on Screenspot-pro:
```bash
python qwen25_vllm_screenspot_pro.py --annotation_path <path_to_annotation(folder containing json files)> --model_path <path_to_model> --image_dir <path_to_image_dir>
```

## Jedi on OSWorld-G
Run the following command to evaluate on OSWorld-G:
```bash
python qwen25_vllm_osworld_g.py --annotation_path <path_to_annotation(json file)> --model_path <path_to_model> --classification_path <path_to_classification_result> --image_dir <path_to_image_dir>
```

## Other open source models on OSWorld-G
You can modify our script to test other open-source models on these benchmark datasets. An example is provided in `qwen2_vllm_osworld_g_aguvis.py`.

Run the following command to evaluate on OSWorld-G using Aguvis:
```bash
python qwen2_vllm_osworld_g_aguvis.py --annotation_path <path_to_annotation> --model_path <path_to_model> --classification_path <path_to_classification_result> --image_dir <path_to_image_dir>
```

## Holo on OSWorld-G
Run the following command to evaluate a Holo model (Holo3.1 / Holo1.5 family) on OSWorld-G. The script starts the vLLM server itself:
```bash
python holo_osworld_g.py --annotation_path ../benchmark/OSWorld-G.json --image_dir ../benchmark/images --classification_path ../benchmark/classification_result.json --model_path <path_to_model> --model_name holo
```

Pass `--model_path` only when you want the script to launch vLLM. To reuse a server you already started, drop it and point at the endpoint with `--base_url`/`--port`.

Holo returns a click point as JSON (`{"x": ..., "y": ...}`) normalized to `[0, 1000]`, which the script scales back to pixels using each screenshot's own dimensions -- no `smart_resize` round-trip is needed. Useful flags:

- `--refusal_type explicit` (default) allows `null` coordinates so the 54 refusal items are answerable; `implicit` uses the documented schema verbatim, which scores 0 on those items.
- `--structured_output {json_schema,guided_json,none}` selects how the JSON is constrained server-side; use `guided_json` on vLLM builds without `response_format` support.
- `--use_cache` reuses responses between runs, `--output_path` writes per-item predictions.

### Running it in Docker

`Dockerfile` at the repository root builds an image with vLLM, the runner and the
benchmark data. Build from the repository root:

```bash
docker build -t osworld-g-holo .
```

Then mount the checkpoint and a results directory:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g -v /models/Holo-3.1-pruned:/models/holo:ro -v "$PWD/results":/results --user "$(id -u):$(id -g)" osworld-g-holo --model_path /models/holo --model_name holo-pruned --use_cache --output_path /results/preds.json
```

The three benchmark paths default to their in-image locations, so only the model
flags are needed. Weights are never baked into the image; mount them at `/models`.
Pass `--build-arg VLLM_TAG=<tag>` at build time to match the vLLM version your
checkpoint needs.

## Other closed source models on OSWorld-G
You can also evaluate other closed-source models on OSWorld-G. An example with Operator is provided in `operator_osworld_g.py`.

Run the following command to evaluate on OSWorld-G using Operator:
```bash
python operator_osworld_g.py --annotation_path <path_to_annotation> --model_name <path_to_model> --classification_path <path_to_classification_result> --image_dir <path_to_image_dir>
```

python operator_osworld_g.py --annotation_path ../benchmark/OSWorld-G.json --model_name kimiv_grounding_sota_1221 --image_dir ../benchmark/images