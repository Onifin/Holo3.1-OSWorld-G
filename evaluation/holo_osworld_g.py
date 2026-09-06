# example usage:
#   python holo_osworld_g.py \
#       --annotation_path ../benchmark/OSWorld-G.json \
#       --image_dir ../benchmark/images \
#       --classification_path ../benchmark/classification_result.json \
#       --model_path Hcompany/Holo1.5-7B --model_name holo
#
# Evaluates a Holo model (Holo3.1 / Holo1.5 family) on OSWorld-G through a local
# vLLM OpenAI-compatible server.
#
# Holo's element-localization interface (https://hub.hcompany.ai/element-localization):
#   * single-turn, no system prompt, temperature 0, enable_thinking=False
#   * output is structured JSON {"x": int, "y": int}, normalized to [0, 1000]
#   * the point is scaled back to pixels with the *same* image bytes that were sent,
#     so no smart_resize round-trip is needed (unlike the Qwen tool-call scripts).
import argparse
import base64
import json
import os
import re
import shlex
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import torch
from loguru import logger as eval_logger
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from eval import GroundingEval

MAX_ATTEMPTS = 5
NUM_SECONDS_TO_SLEEP = 5

# The click-point schema from the Holo docs. `x` / `y` are integers in [0, 1000],
# normalized to the image.
LOCALIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}

# OSWorld-G contains 54 refusal items (no element matches the instruction). The
# documented schema has no way to express "not present", so for the explicit
# refusal setting we widen `x` / `y` to accept null and say so in the prompt.
REFUSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": ["integer", "null"], "minimum": 0, "maximum": 1000},
        "y": {"type": ["integer", "null"], "minimum": 0, "maximum": 1000},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}

LOCALIZATION_PROMPT = (
    "Localize an element on the GUI image according to the provided target and output a click position.\n"
    " * You must output a valid JSON following the format: {schema}\n"
    " Your target is:\n{element}"
)

REFUSAL_CLAUSE = (
    " * If no element on the image corresponds to the target, output null for both x and y.\n"
)

# `explicit` bolts the refusal clause onto a prompt whose first sentence already
# takes the element's existence for granted, and the model never once used the
# null. This variant makes presence the first decision instead of an aside: the
# opening sentence is conditional, the absence case leads the bullets, and
# guessing the nearest match is ruled out by name. Same schema, same output
# format, same cadence as the documented prompt -- only the framing moves.
CONDITIONAL_PROMPT = (
    "Decide whether the target element is present on the GUI image, and output a "
    "click position only if it is.\n"
    " * The target may not be present at all. If no element on the image "
    "corresponds to the target, output null for both x and y -- do not fall back "
    "to the closest or most similar element.\n"
    " * You must output a valid JSON following the format: {schema}\n"
    " Your target is:\n{element}"
)


def build_prompt(instruction, refusal_type):
    """Return the prompt and the schema the server should constrain against."""
    if refusal_type == "conditional":
        schema = REFUSAL_SCHEMA
        return CONDITIONAL_PROMPT.format(
            schema=json.dumps(schema), element=instruction
        ), schema

    schema = REFUSAL_SCHEMA if refusal_type == "explicit" else LOCALIZATION_SCHEMA
    prompt = LOCALIZATION_PROMPT.format(
        schema=json.dumps(schema), element=instruction
    )
    if refusal_type == "explicit":
        # keep the refusal clause next to the other bulleted constraint
        prompt = prompt.replace(" Your target is:", REFUSAL_CLAUSE + " Your target is:")
    return prompt, schema


def parse_coordinates(response, image_size):
    """Turn a Holo response into [x1, y1, x2, y2] pixel coordinates.

    Returns [-1, -1, -1, -1] for an explicit refusal, which GroundingEval scores as
    correct on `refusal` items and incorrect everywhere else. An unparseable
    response returns [0, 0, 0, 0] instead, so a broken generation is never
    credited as a refusal.
    """
    width, height = image_size
    try:
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            )
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("no JSON object in response")
        data = json.loads(match.group(0))

        x, y = data.get("x"), data.get("y")
        if x is None or y is None:
            return [-1, -1, -1, -1]

        abs_x = float(x) / 1000 * width
        abs_y = float(y) / 1000 * height
        return [abs_x, abs_y, abs_x, abs_y]
    except Exception:
        eval_logger.warning(f"Error parsing coordinates from: {response!r}")
        return [0, 0, 0, 0]


class HoloVLLM:
    """Thin client over the vLLM OpenAI-compatible endpoint."""

    def __init__(self, model_name, base_url, api_key, refusal_type,
                 structured_output, max_workers, max_tokens):
        self.model_name = model_name
        self.refusal_type = refusal_type
        self.structured_output = structured_output
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _request_kwargs(self, schema):
        # enable_thinking=False is the documented single-turn setting for Holo.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        kwargs = {}
        if self.structured_output == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "click_position",
                    "schema": schema,
                    "strict": True,
                },
            }
        elif self.structured_output == "guided_json":
            extra_body["guided_json"] = schema
        kwargs["extra_body"] = extra_body
        return kwargs

    def generate_until(self, instances) -> List[str]:
        res = [None] * len(instances)
        pbar = tqdm(total=len(instances), desc="Model Responding")

        def process_request(index_instance):
            index, instance = index_instance
            prompt, schema = build_prompt(instance["instruction"], self.refusal_type)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{instance['image_b64']}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            response_text = ""
            for attempt in range(MAX_ATTEMPTS):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                        **self._request_kwargs(schema),
                    )
                    response_text = completion.choices[0].message.content or ""
                    break
                except Exception as e:
                    eval_logger.error(f"Error during API call: {e}")
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(f"All {MAX_ATTEMPTS} attempts failed.")

            pbar.update(1)
            return index, response_text

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(process_request, (i, instance))
                for i, instance in enumerate(instances)
            ]
            for future in as_completed(futures):
                index, response_text = future.result()
                res[index] = response_text

        pbar.close()
        return res

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for this model.")


class BenchmarkRunner:
    def __init__(self, annotation_path, image_dir, classification_path, model,
                 use_cache=False, output_path=None, cache_dir="."):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.classification_path = classification_path
        self.model = model
        self.use_cache = use_cache
        self.output_path = output_path
        self.cache_dir = cache_dir

    def load_annotations(self):
        with open(self.annotation_path, "r") as f:
            data = json.load(f)

        flatten_data_items = []
        for i, item in enumerate(data):
            image_path = os.path.join(self.image_dir, item["image_path"])
            with Image.open(image_path) as image:
                image_size = [image.width, image.height]
            # Send the file's own bytes: any re-encode risks a coordinate mismatch.
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            flatten_data_items.append(
                {
                    "id": image_path[:-4],
                    "annotation_id": str(i),
                    "data_id": item["id"],
                    "image_b64": image_b64,
                    "image_path": item["image_path"],
                    "instruction": item["instruction"],
                    "image_size": image_size,
                    "box_type": item["box_type"],
                    "box_coordinates": item["box_coordinates"],
                }
            )

        return flatten_data_items

    def _cache_file(self):
        name = (
            self.model.model_name.replace("/", "_")
            + self.annotation_path.replace("/", "_").replace(".json", ".cache")
            + f"_{self.model.refusal_type}_prediction_cache.json"
        )
        return os.path.join(self.cache_dir, name)

    def evaluate(self):
        items = self.load_annotations()
        evaluator = GroundingEval(None)

        predictions_cache = {}
        cache_file = self._cache_file()
        if self.use_cache and os.path.exists(cache_file):
            print("Loading cache file: ", cache_file)
            with open(cache_file, "r") as f:
                predictions_cache = json.load(f)

        with open(self.classification_path, "r") as f:
            classification_result = json.load(f)

        accuracy_dict_group = {}
        instances = []
        for item in items:
            instance_group_list = []
            item["instance_id"] = f"{item['id']}_{item['annotation_id']}"
            for cls_type, classification_items in classification_result[
                "classified"
            ].items():
                for classification_item in classification_items:
                    if classification_item["id"] == item["data_id"]:
                        instance_group_list.append(cls_type)
                        break
            if len(instance_group_list) == 0:
                instance_group_list.append("unclassified")
            item["instance_group_list"] = instance_group_list
            for instance_group in instance_group_list:
                if instance_group not in accuracy_dict_group:
                    accuracy_dict_group[instance_group] = {
                        "total": 0,
                        "correct": 0,
                        "accuracy": 0,
                    }
                accuracy_dict_group[instance_group]["total"] += 1

            if item["instance_id"] not in predictions_cache:
                instances.append(item)

        if instances:
            responses = self.model.generate_until(instances)
            for instance, response in zip(instances, responses):
                predictions_cache[instance["instance_id"]] = (response or "").strip()
            if self.use_cache:
                os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(predictions_cache, f)

        total = len(items)
        correct = 0
        records = []

        for item in items:
            response = predictions_cache.get(item["instance_id"], "")
            predicted_coords = parse_coordinates(response, item["image_size"])

            image_size = item["image_size"]
            if "bbox" == item["box_type"]:
                boxes_type = "bbox"
                boxes_coordinate = item["box_coordinates"][:2]
                boxes_size = item["box_coordinates"][2:]
            elif "polygon" == item["box_type"]:
                boxes_type = "polygon"
                boxes_coordinate = item["box_coordinates"]
                boxes_size = image_size
            elif "refusal" == item["box_type"]:
                boxes_type = "refusal"
                boxes_coordinate = item["box_coordinates"]
                boxes_size = image_size
            else:
                raise ValueError(f"Unknown box type: {item['box_type']}")

            is_correct = evaluator._eval(
                predicted_coords, boxes_type, boxes_size, boxes_coordinate, image_size
            )

            if is_correct:
                correct += 1
                for instance_group in item["instance_group_list"]:
                    accuracy_dict_group[instance_group]["correct"] += 1

            records.append(
                {
                    "data_id": item["data_id"],
                    "image_path": item["image_path"],
                    "instruction": item["instruction"],
                    "box_type": item["box_type"],
                    "response": response,
                    "predicted_point": predicted_coords[:2],
                    "correct": bool(is_correct),
                    "groups": item["instance_group_list"],
                }
            )

        accuracy = correct / total
        for group in accuracy_dict_group:
            stats = accuracy_dict_group[group]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

        results = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "cached_predictions": len(predictions_cache),
            "accuracy_dict_group": accuracy_dict_group,
        }

        if self.output_path:
            with open(self.output_path, "w") as f:
                json.dump({"results": results, "predictions": records}, f, indent=2)
            print(f"Wrote per-item predictions to {self.output_path}")

        return results


def start_vllm_service(ckpt_path, port, model_name, max_model_len,
                       max_num_seqs, extra=""):
    command = [
        "vllm",
        "serve",
        ckpt_path,
        "--served-model-name",
        model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(torch.cuda.device_count()),
        "--max-model-len",
        str(max_model_len),
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
        # The runner never has more than --max_workers requests in flight, so
        # the server's default batch width is pure waste -- and on the hybrid
        # Qwen3.5-MoE checkpoints it is fatal: every decode sequence holds a
        # Mamba state block, and asking for 1024 of them exceeds what fits,
        # which aborts CUDA graph capture before the server ever listens.
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    # Anything else the checkpoint needs -- --gpu-memory-utilization above all,
    # since 70 GB of bf16 weights leave little room under the 0.9 default.
    command += shlex.split(extra)
    print("[vllm] " + " ".join(command), flush=True)
    return subprocess.Popen(command)


def wait_for_service(port, timeout=1800):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", port)) == 0:
                return True
        time.sleep(1)
    return False


def terminate_vllm_service(process):
    try:
        process.terminate()
        time.sleep(5)
        if process.poll() is None:
            process.kill()
            time.sleep(1)
    except Exception as e:
        print(f"Failed to terminate VLLM service: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Holo model on OSWorld-G via a local vLLM server."
    )
    parser.add_argument("--annotation_path", type=str,
                        default=os.environ.get("ANNOTATION_PATH", "../benchmark/OSWorld-G.json"),
                        help="Path to the annotation file (default: ../benchmark/OSWorld-G.json).")
    parser.add_argument("--image_dir", type=str,
                        default=os.environ.get("IMAGE_DIR", "../benchmark/images"),
                        help="Directory containing the benchmark images (default: ../benchmark/images).")
    parser.add_argument("--classification_path", type=str,
                        default=os.environ.get("CLASSIFICATION_PATH", "../benchmark/classification_result.json"),
                        help="Path to the classification result file "
                             "(default: ../benchmark/classification_result.json).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model checkpoint to serve. Omit to reuse an already running server.")
    parser.add_argument("--model_name", type=str, default="holo",
                        help="Served model name (default: 'holo').")
    parser.add_argument("--port", type=int, default=8908,
                        help="Port for the vLLM service (default: 8908).")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Override the OpenAI base_url (default: http://localhost:<port>/v1).")
    parser.add_argument("--api_key", type=str, default="token-abc123")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--vllm_extra", type=str,
                        default=os.environ.get("VLLM_EXTRA", ""),
                        help="Extra flags forwarded verbatim to `vllm serve`, "
                             "e.g. --vllm_extra \"--gpu-memory-utilization 0.95\".")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Concurrent requests to the server (default: 8).")
    parser.add_argument("--max_num_seqs", type=int, default=64,
                        help="Server-side batch width, forwarded to vllm serve "
                             "(default: 64, comfortably above --max_workers).")
    parser.add_argument("--refusal_type", type=str, default="explicit",
                        choices=["explicit", "implicit", "conditional"],
                        help="'explicit' appends a refusal clause to the documented prompt; "
                             "'conditional' reframes it so presence is decided first; "
                             "'implicit' uses the documented prompt and schema verbatim, "
                             "which scores 0 on the refusal items (default: explicit).")
    parser.add_argument("--structured_output", type=str, default="json_schema",
                        choices=["json_schema", "guided_json", "none"],
                        help="How to constrain the JSON output on the server side.")
    parser.add_argument("--use_cache", action="store_true",
                        help="Reuse cached model responses between runs.")
    parser.add_argument("--cache_dir", type=str,
                        default=os.environ.get("CACHE_DIR", "."),
                        help="Where the response cache is written (default: cwd).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to write per-item predictions (JSON).")
    args = parser.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}/v1"

    process = None
    if args.model_path:
        process = start_vllm_service(
            args.model_path, args.port, args.model_name, args.max_model_len,
            args.max_num_seqs, args.vllm_extra,
        )
        if not wait_for_service(args.port):
            print(f"Failed to start VLLM service on port {args.port}")
            terminate_vllm_service(process)
            raise SystemExit(1)

    try:
        model = HoloVLLM(
            model_name=args.model_name,
            base_url=base_url,
            api_key=args.api_key,
            refusal_type=args.refusal_type,
            structured_output=args.structured_output,
            max_workers=args.max_workers,
            max_tokens=args.max_tokens,
        )
        runner = BenchmarkRunner(
            annotation_path=args.annotation_path,
            image_dir=args.image_dir,
            classification_path=args.classification_path,
            model=model,
            use_cache=args.use_cache,
            output_path=args.output_path,
            cache_dir=args.cache_dir,
        )

        results = runner.evaluate()
        print("Evaluation Results:")
        print(f"Total samples: {results['total']}")
        print(f"Correct predictions: {results['correct']}")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print("Accuracy by Group:")
        for group, stats in results["accuracy_dict_group"].items():
            print(
                f"  {group}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})"
            )
    finally:
        if process is not None:
            terminate_vllm_service(process)
