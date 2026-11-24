"""Author public CC0 smoke records without reading user datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

C_EXAMPLES = (
    ("add", "int add(int left, int right) {\n    int result = left + right;\n    return result;\n}\n"),
    ("maximum", "int maximum(int left, int right) {\n    if (left > right) { return left; }\n    return right;\n}\n"),
    ("minimum", "int minimum(int left, int right) {\n    if (left < right) { return left; }\n    return right;\n}\n"),
    ("absolute", "int absolute(int value) {\n    if (value < 0) { return -value; }\n    return value;\n}\n"),
    ("sum", "int sum(const int *values, int count) {\n    int total = 0;\n    for (int i = 0; i < count; ++i) { total += values[i]; }\n    return total;\n}\n"),
    ("count_even", "int count_even(const int *values, int count) {\n    int total = 0;\n    for (int i = 0; i < count; ++i) { if (values[i] % 2 == 0) { ++total; } }\n    return total;\n}\n"),
    ("factorial", "int factorial(int value) {\n    int result = 1;\n    for (int i = 2; i <= value; ++i) { result *= i; }\n    return result;\n}\n"),
    ("clamp", "int clamp(int value, int lower, int upper) {\n    if (value < lower) { return lower; }\n    if (value > upper) { return upper; }\n    return value;\n}\n"),
    ("contains", "int contains(const int *values, int count, int target) {\n    for (int i = 0; i < count; ++i) { if (values[i] == target) { return 1; } }\n    return 0;\n}\n"),
    ("reverse", "void reverse(int *values, int count) {\n    for (int i = 0; i < count / 2; ++i) {\n        int saved = values[i];\n        values[i] = values[count - i - 1];\n        values[count - i - 1] = saved;\n    }\n}\n"),
    ("copy", "void copy(int *output, const int *input, int count) {\n    for (int i = 0; i < count; ++i) { output[i] = input[i]; }\n}\n"),
    ("all_positive", "int all_positive(const int *values, int count) {\n    for (int i = 0; i < count; ++i) { if (values[i] <= 0) { return 0; } }\n    return 1;\n}\n"),
)
CHAT_EXAMPLES = (
    ("What is two plus three?", "Two plus three is five."),
    ("What does a compiler do?", "A compiler translates source code into another representation."),
    ("What is a local model?", "A local model runs on your own computer."),
    ("What does a unit test check?", "A unit test checks one small piece of behavior."),
    ("What is a repository?", "A repository stores a project's files and change history."),
    ("Why keep a fixed seed?", "A fixed seed helps reproduce an experiment."),
    ("What is a validation split?", "A validation split holds examples out of training."),
    ("What is a patch?", "A patch describes changes to files."),
    ("What is a tokenizer?", "A tokenizer converts text into token identifiers."),
    ("What is retrieval?", "Retrieval finds relevant stored context for a request."),
    ("Why review a change?", "Review helps catch mistakes before a change is accepted."),
    ("What happens after a failed test?", "Inspect the failure, propose a correction, and test again."),
)


def write_fixtures(out_dir: str | Path, *, kind: str = "c") -> list[Path]:
    if kind not in {"c", "chat", "sft"}:
        raise ValueError("kind must be c, chat or sft")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    if kind == "sft":
        paths = [root / "train.jsonl", root / "validation.jsonl"]
        for path, examples in zip(paths, (CHAT_EXAMPLES[:8], CHAT_EXAMPLES[8:])):
            rows = [{"prompt": prompt, "response": response, "license": "CC0-1.0", "consent": True}
                    for prompt, response in examples]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return paths
    paths = []
    examples = C_EXAMPLES if kind == "c" else CHAT_EXAMPLES
    for index, (name, text) in enumerate(examples):
        record = {"license": "CC0-1.0", "consent": True, "group_id": f"public-{kind}-{index}"}
        if kind == "c":
            record["code"] = f"/* Public synthetic example: {name}. This fixture demonstrates a small C function. */\n{text}"
        else:
            record["messages"] = [
                {"role": "system", "content": "Give a short answer."},
                {"role": "user", "content": name},
                {"role": "assistant", "content": text},
            ]
        path = root / f"example-{index:02d}.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are refused.")
    parser.add_argument("--kind", choices=("c", "chat", "sft"), default="c")
    args = parser.parse_args()
    paths = write_fixtures(args.out, kind=args.kind)
    print(f"Wrote {len(paths)} authored CC0 fixture files with distinct split groups to {args.out}")


if __name__ == "__main__":
    main()

