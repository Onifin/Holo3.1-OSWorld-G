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

`benchmark/app_labels.json` adds one. Every one of the 251 screenshots was
labelled by reading the application name out of the GNOME top bar in the image
itself, so these are read labels rather than keywords guessed from the
instruction text. The categories are the eight OSWorld domains, and `null` for
items in applications outside that set:

| category | items | screenshots |
| --- | ---: | ---: |
| Chrome | 94 | 51 |
| LibreOffice Impress | 90 | 31 |
| LibreOffice Calc | 81 | 29 |
| GIMP | 77 | 29 |
| LibreOffice Writer | 42 | 23 |
| OS | 41 | 19 |
| VsCode | 39 | 21 |
| VLC | 38 | 19 |
| `null` | 62 | 29 |
| **total** | **564** | **251** |

`OS` covers the desktop environment itself: terminal (13), file manager (10),
settings (10), the Activities overview (6) and the login screen (2). `null`
covers Thunderbird (24), Evince (20), Gedit (6), Totem (5), Ubuntu Software (3),
Image Viewer (3) and the LibreOffice Start Center (1) -- note that Thunderbird is
a domain of its own in OSWorld, so promote it to a category of its own if you
are comparing against numbers from there.

`accuracy_by_app.py` joins those labels with the `*_preds.json` a run writes:

```bash
python accuracy_by_app.py /results/dense_preds.json --exclude_refusal
```

Pass two files to get them side by side with a per-application delta, which is
the view worth having when comparing a pruned checkpoint against its baseline:

```bash
python accuracy_by_app.py /results/dense_preds.json /results/pruned_preds.json --exclude_refusal
```

`--exclude_refusal` drops the 54 refusal items. They are worth dropping from a
comparison: answering one correctly requires the model to decline to point at
anything, and grounding-specialised models simply do not -- Holo-3.1-35B-A3B
scores 0/54, emitting a valid coordinate every time, with or without a refusal
clause in the prompt and with or without constrained decoding.

## Other closed source models on OSWorld-G
You can also evaluate other closed-source models on OSWorld-G. An example with Operator is provided in `operator_osworld_g.py`.

Run the following command to evaluate on OSWorld-G using Operator:
```bash
python operator_osworld_g.py --annotation_path <path_to_annotation> --model_name <path_to_model> --classification_path <path_to_classification_result> --image_dir <path_to_image_dir>
```

python operator_osworld_g.py --annotation_path ../benchmark/OSWorld-G.json --model_name kimiv_grounding_sota_1221 --image_dir ../benchmark/images