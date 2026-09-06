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

## Breaking a run down by application

OSWorld-G groups its items by GUI element type -- `benchmark/buckets.json` maps
`Label` to `text_matching`, `Icon`/`Image`/`Button` to `element_recognition`, and
so on -- and the annotations carry no field saying which application is on screen.

`benchmark/task_labels.json` adds one, per task rather than per screenshot. The
per-screenshot shortcut does not survive contact with the data: a screenshot can
hold two applications -- GIMP full-screen with the Settings window over it, a
Writer document behind a Chrome window -- and the tasks on it then belong to
different ones. Every one of the 564 tasks was reviewed by hand with its
instruction and its target box drawn on the screenshot, which is what settles
those. The categories are the eight OSWorld domains:

| category | tasks |
| --- | ---: |
| Chrome | 75 |
| LibreOffice Impress | 85 |
| LibreOffice Calc | 77 |
| GIMP | 76 |
| LibreOffice Writer | 36 |
| OS | 54 |
| VsCode | 36 |
| VLC | 29 |
| `null` | 96 |
| **total** | **564** |

`OS` is the desktop environment itself -- terminal, file manager, settings, the
dock and the Activities overview -- including tasks whose target is the dock icon
of an application rather than the application. `null` is everything outside the
eight: Thunderbird, Evince, Gedit, Totem and friends, plus all 54
refusal tasks, since nothing on screen answers those. That last part is worth
knowing: it leaves the eight real categories with a ceiling of 100%, so
`--exclude_refusal` is optional rather than necessary when reading them.

`validate_labels.py` is the review tool those labels came out of, and the way to
revise them. It shows one task per screen -- instruction, target box, top bar at
native resolution, current label -- and you press `a` to approve or `1`..`8` /
`0` to replace; `A` applies the label you just chose to every task on that
screenshot. It reads and rewrites `task_labels.json`, records what you have seen
in `task_labels_review.json`, backs both up to `.bak` on save, and reopens at the
first task with no verdict:

```bash
uv run --with opencv-python --with numpy validate_labels.py
```

Add `--only null` to review just the unlabelled tasks, or `--only GIMP` to sweep a
single category.

`accuracy_by_app.py` joins those labels with the `*_preds.json` a run writes:

```bash
python accuracy_by_app.py /results/dense_preds.json
```

Pass two files to get them side by side with a per-application delta, which is
the view worth having when comparing a pruned checkpoint against its baseline:

```bash
python accuracy_by_app.py /results/dense_preds.json /results/pruned_preds.json
```

`--exclude_refusal` drops the 54 refusal tasks entirely. They all sit under
`null`, so the eight categories are unaffected either way; what the flag changes
is the total. Worth knowing about them: answering one correctly requires the
model to decline to point at anything, and grounding-specialised models simply do
not -- Holo-3.1-35B-A3B scores 0/54, emitting a valid coordinate every time, with
or without a refusal clause in the prompt and with or without constrained
decoding.

## Other closed source models on OSWorld-G
You can also evaluate other closed-source models on OSWorld-G. An example with Operator is provided in `operator_osworld_g.py`.

Run the following command to evaluate on OSWorld-G using Operator:
```bash
python operator_osworld_g.py --annotation_path <path_to_annotation> --model_name <path_to_model> --classification_path <path_to_classification_result> --image_dir <path_to_image_dir>
```

python operator_osworld_g.py --annotation_path ../benchmark/OSWorld-G.json --model_name kimiv_grounding_sota_1221 --image_dir ../benchmark/images