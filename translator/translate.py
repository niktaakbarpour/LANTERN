import os
import time
import tqdm
import json
# import openai
# import requests
# from openai import OpenAI
import argparse
import datasets
import concurrent
import numpy as np
from promptsource.templates import Template
from analyzer.decision import TransDecision
from middleware.retrieval import build_target_db
from middleware.history import get_last_incorrect_samples, cp_last_incorrect_samples
from middleware import prompt
import shutil
from middleware.deepseek_local import Message

SHORT_LANG_MAP = {
    "GNU C++": "C++",
    "GNU C++17": "C++",
    "MS C++ 2017": "C++",
    "MS C++": "C++",
    "Java 8": "Java",
    "Java 6": "Java",
    "GNU C++11": "C++",
    "Java 11": "Java",
    "GNU C++14": "C++",
    "Mono C#": "C#",
    "GNU C": "C",
    "Python 3": "Python",
    "PyPy 3": "Python",
    "GNU C11": "C",
    "Go": "Go",
    "Rust": "Rust",
    "PyPy 2": "Python",
    "Python 2": "Python",
    "MS C#": "C#",
    "Kotlin": "Kotlin",
    "GNU C++0x": "C++",
    "Java 7": "Java",
    "Node.js": "Javascript",
    ".NET Core C#": "C#",
    "PHP": "PHP",
    "GNU C++17 Diagnostics": "C++",
    "Clang++17 Diagnostics": "C++",
    "JavaScript": "Javascript",
    "Ruby": "Ruby",
    "C# 10": "C#",
    "C# 8": "C#",
    "Clang++20 Diagnostics": "C++",
    "GNU C++17 (64)": "C++",
    "GNU C++20 (64)": "C++",
    "Java 17": "Java",
    "Kotlin 1.4": "Kotlin",
    "Kotlin 1.5": "Kotlin",
    "Kotlin 1.6": "Kotlin",
    "Kotlin 1.7": "Kotlin",
    "PyPy 3-64": "Python",
    "Python 3 + libs": "Python",
    "Ruby 3": "Ruby",
    "Rust 2021": "Rust",
}

LANGS = sorted(set([v for k, v in SHORT_LANG_MAP.items()]))


# openai.api_key = os.environ["API_KEY"]
# openai.api_base = os.environ["API_BASE"]
# model_name = os.environ["MODEL_NAME"]

def gen(prompt_text, temperature, nsample, llm):
    cnt = 0
    while cnt < 999:
        try:
            messages = [Message(role="user", content=prompt_text)]
            prompt = llm.prepare_prompt(messages)
            full_prompt = prompt + "Assistant:"
            tokens = llm.tokenizer.encode(full_prompt, return_tensors="pt").to(llm.model.device)
            outputs = llm.model.generate(
                tokens,
                max_new_tokens=512,
                # do_sample=True,
                temperature=temperature,
                top_p=1.0,
                eos_token_id=llm.tokenizer.encode("<|end▁of▁sentence|>")[0]
            )
            print("get deepseek response......")
            break
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"Gen Error: {e}")
    else:
        return None

    raw = llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = raw.split("<|end▁of▁sentence|>")[0].strip()
    content = llm.extract_output(raw)
    temp = {"choices": [{"message": {"content": content}}], "prompt": prompt_text}
    return {"choices": [{"message": {"content": content}}], "prompt": prompt_text}

# Alias preserving interface
def gen_request(prompt_text, temperature, nsample, llm):
    res = gen(prompt_text, temperature, nsample, llm)
    if res is None:
        return None
    return {"data": [{"content": res['choices'][0]['message']['content'], "type": "text"}], "prompt": prompt_text}



def process_prompt(dt, temperature, trans_dir, target_lang, index, r_mode, llm, dry_run=0):
    dt["source_lang"] = dt["lang_cluster"]
    dt["target_lang"] = target_lang
    language = f"{dt['source_lang']}--{dt['target_lang']}"
    file_path = os.path.join(trans_dir, f"{index}_{temperature}_{language}.json")
    if not os.path.exists(file_path):
        if r_mode != "ultimate2":
            dt["prob_desc_sample_inputs"] = json.loads(dt["prob_desc_sample_inputs"])
            dt["prob_desc_sample_outputs"] = json.loads(dt["prob_desc_sample_outputs"])
        lm_io = prompt.trans(dt)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
        else:
            out = gen(lm_io[0], temperature, 1, llm)
            # out = gen_request(lm_io[0], temperature, 1)
            export_data = {"oai_response": out, "source_data": dt}
            open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")


def run(base_dir, num_proc, dry_run, it, mode, r_mode, dataset_path, llm, config_path=""):
    iter_dir = os.path.join(base_dir, f"iter_{it}")
    unfixed_file = os.path.join(iter_dir, "unfixed.json")
    if os.path.exists(unfixed_file):
        with open(unfixed_file, 'r') as f:
            unfixed_ids = list(json.load(f).keys())
    else:
        raise FileNotFoundError("unfixed.json not found")

    trans_dir = os.path.join(iter_dir, "trans")
    os.makedirs(trans_dir, exist_ok=True)

    # copy mode
    if mode == "copy":
        cp_last_incorrect_samples(base_dir, it, unfixed_ids)
        return

    # decision-based translation
    decision = TransDecision(base_dir, it, config_path)
    if mode in ["reasoning", "nohist", "nocot"]:
        build_target_db(base_dir, it)

    apr_dataset = datasets.load_from_disk(dataset_path)
    # langs = ["Ruby"]
    # apr_dataset = apr_dataset.filter(lambda example: example["lang_cluster"] in langs)

    first_entry = apr_dataset.select(range(10))
    if r_mode == "ultimate2":
        unfixed_dataset = get_last_incorrect_samples(base_dir, it, unfixed_ids)
    else:
        unfixed_dataset = apr_dataset.filter(lambda x: x['bug_code_uid'] in unfixed_ids)

    temperature_list = [0.3157894736842105]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_proc)) as executor:
        futures = []
        for idx, dt in enumerate(tqdm.tqdm(unfixed_dataset, desc="Preparing samples lang")):
            target_lang = decision.decide_lang(sample=dt, it=it, mode=mode)
            for temp in temperature_list:
                process_prompt(dt, temp, trans_dir, target_lang, idx, r_mode, llm, dry_run)
        #         futures.append(
        #             executor.submit(
        #                 process_prompt,
        #                 dt, temp, trans_dir, target_lang, idx, r_mode, llm, dry_run
        #             )
        #         )
        # for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Translating"):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="dumped/oai/apr_n_sample_20",
        help="Path to the trans-repair base directory.",
    )
    parser.add_argument(
        "--num-proc",
        default=1,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--dry-run",
        default=0,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--it",
        default=1,
        type=int,
        help="Current iteration epoch of trans-repair.",
    )
    parser.add_argument(
        "--mode",
        default="vanilla",
        help="Translation mode.",
    )
    parser.add_argument(
        "--r_mode",
        default="vanilla",
        help="Repair mode.",
    )
    args = parser.parse_args()
    run(args.base_dir, args.num_proc, args.dry_run, args.it, args.mode, args.r_mode)
