import random
import json
from pathlib import Path
from argparse import ArgumentParser


NO_INPUT_PROMPT: str = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"


def main():
    parser = ArgumentParser()
    parser.add_argument("--ichikara-dir", type=str, required=True)
    parser.add_argument("--answer-carefully-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    ichikara_filenames: list[str] = [
        "ichikara-instruction-003-001-1",
        "ichikara-instruction-003-003-1",
    ]
    saved_ichikara_samples: list[dict] = []
    for ichikara_filename in ichikara_filenames:
        ichikara_filepath: Path = Path(f"{args.ichikara_dir}/{ichikara_filename}.json")
        print(ichikara_filepath)
        with ichikara_filepath.open(mode="r", encoding="utf-8") as f:
            loaded_samples = json.load(f)
        for loaded_sample in loaded_samples:
            saved_ichikara_samples.append(
                {
                    "ID": loaded_sample["ID"],
                    "messages": [
                        {"role": "system", "content": NO_INPUT_PROMPT},
                        {"role": "user", "content": loaded_sample["text"]},
                        {"role": "assistant", "content": loaded_sample["output"]},
                    ],
                }
            )

    random.seed(42)
    random.shuffle(saved_ichikara_samples)
    with Path(f"{args.output_dir}/ichikara.jsonl").open("w", encoding="utf-8") as f:
        for sample in saved_ichikara_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    answer_carefully_filepath: Path = Path(f"{args.answer_carefully_dir}/AnswerCarefullyVersion002_Dev.json")
    with answer_carefully_filepath.open(mode="r", encoding="utf-8") as f:
        loaded_samples = json.load(f)
    saved_answer_carefully_samples: list[dict] = []
    for loaded_sample in loaded_samples:
        saved_answer_carefully_samples.append(
            {
                "ID": loaded_sample["ID"],
                "messages": [
                    {"role": "system", "content": NO_INPUT_PROMPT},
                    {"role": "user", "content": loaded_sample["text"]},
                    {"role": "assistant", "content": loaded_sample["output"]},
                ],
            }
        )

    random.seed(42)
    random.shuffle(saved_answer_carefully_samples)
    with Path(f"{args.output_dir}/answer_carefully.jsonl").open("w", encoding="utf-8") as f:
        for sample in saved_answer_carefully_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")




if __name__ == "__main__":
    main()

