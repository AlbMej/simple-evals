import json
import os

import pandas as pd

from . import common
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o_chat_completion_sampler import OChatCompletionSampler


def main():
    debug = True  # Set to False to run on full MMLU set for specified language
    # Set environment variables to point to your local vLLM server
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
    os.environ["OPENAI_API_KEY"] = "dummy-key"  # An api key is required

    samplers = {
        "local_gemma3_270m": ChatCompletionSampler(
            model="unsloth/gemma-3-270m-it",
        ),
        # "local_qwen2": ChatCompletionSampler(
        #     model="Qwen/Qwen2-0.5B-Instruct",  # model identifier for the API server
        # ),
        # "local_oss20b": ChatCompletionSampler(
        #     model="gpt-oss-20b",
        # ),

    }

    def get_evals(eval_name):
        match eval_name:
            case "mmlu_EN-US":
                return MMLUEval(num_examples=10 if debug else None, language="EN-US")
            case "mmlu_AR-XY":
                return MMLUEval(num_examples=10 if debug else None, language="AR-XY")
            case "mmlu_BN-BD":
                return MMLUEval(num_examples=10 if debug else None, language="BN-BD")
            case "mmlu_DE-DE":
                return MMLUEval(num_examples=10 if debug else None, language="DE-DE")
            case "mmlu_ES-LA":
                return MMLUEval(num_examples=10 if debug else None, language="ES-LA")
            case "mmlu_FR-FR":
                return MMLUEval(num_examples=10 if debug else None, language="FR-FR")
            case "mmlu_HI-IN":
                return MMLUEval(num_examples=10 if debug else None, language="HI-IN")
            case "mmlu_ID-ID":
                return MMLUEval(num_examples=10 if debug else None, language="ID-ID")
            case "mmlu_IT-IT":
                return MMLUEval(num_examples=10 if debug else None, language="IT-IT")
            case "mmlu_JA-JP":
                return MMLUEval(num_examples=10 if debug else None, language="JA-JP")
            case "mmlu_KO-KR":
                return MMLUEval(num_examples=10 if debug else None, language="KO-KR")
            case "mmlu_PT-BR":
                return MMLUEval(num_examples=10 if debug else None, language="PT-BR")
            case "mmlu_ZH-CN":
                return MMLUEval(num_examples=10 if debug else None, language="ZH-CN")
            case "mmlu_SW-KE":
                return MMLUEval(num_examples=10 if debug else None, language="SW-KE")
            case "mmlu_YO-NG":
                return MMLUEval(num_examples=10 if debug else None, language="YO-NG")
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name)
        for eval_name in [
            "mmlu_ES-LA",
        ]
    }
    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}

    results_dir = "mmlu_results"  # For Google Collab, use "/content/drive/MyDrive/FOLDER_NAME"
    os.makedirs(results_dir, exist_ok=True)
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"

            report_filename = f"{results_dir}/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            
            result_filename = f"{results_dir}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
