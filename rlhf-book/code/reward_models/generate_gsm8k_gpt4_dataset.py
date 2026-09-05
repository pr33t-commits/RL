#!/usr/bin/env python3
"""Generate GSM8K reward-model data from GPT-4 solution samples.

For each GSM8K problem this script samples one or more independent solutions
from GPT-4.  It then asks a separate GPT-4 call to judge each solution against
the GSM8K reference answer.  The complete generated solution is retained;
this script never fabricates a negative example by modifying a correct answer.

The output is JSONL.  Each row contains both objective correctness (whether
the candidate's final numeric answer matches GSM8K) and the judge's label.
Keeping both makes it possible to measure generation accuracy and judge
agreement before using the data to train a reward model. Exact numeric matches
are labeled locally; only mismatches are sent to the semantic answer judge.
The judge does not inspect or evaluate reasoning.

Usage:
    uv run python -m reward_models.generate_gsm8k_gpt4_dataset \
        --samples 1000 --solutions-per-problem 2 \
        --output reward_models/data/gsm8k_gpt4.jsonl

Requires ``OPENAI_API_KEY`` in the environment.  Install the OpenAI Python
client if it is not already present: ``pip install openai``.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI


GENERATION_SYSTEM_PROMPT = """You solve GSM8K grade-school math problems.
Work the problem independently and show all of your reasoning in the answer.
End with a separate line in exactly this form: #### <integer answer>
Do not copy or refer to a reference solution; the goal is to produce an
independent sampled attempt, which may occasionally contain a mistake."""

JUDGE_SYSTEM_PROMPT = """You compare two final answers to a GSM8K problem.
Judge only whether the candidate answer is semantically equal to the reference
numeric answer. Do not inspect, request, or evaluate any solution reasoning.
Return only this JSON object, with no markdown fences or extra text:
{"correct": true}"""


def parse_numeric_answer(text: str) -> int | None:
    """Extract the final integer answer from GSM8K-style solution text."""
    matches = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if not matches:
        # Also handle common model variants such as ``boxed{42}`` or a final
        # sentence that does not use GSM8K's delimiter.
        matches = re.findall(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}", text)
    if not matches:
        matches = re.findall(r"(?i)(?:answer|result)\D+(-?\d[\d,]*)", text)
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def gold_numeric_answer(reference: str) -> int:
    """Parse GSM8K's authoritative ``####`` answer."""
    answer = parse_numeric_answer(reference)
    if answer is None:
        raise ValueError(f"Could not parse GSM8K reference answer: {reference!r}")
    return answer


def answer_after_delimiter(text: str) -> str:
    """Return the text following the final GSM8K ``####`` delimiter."""
    if "####" not in text:
        return ""
    return text.rsplit("####", 1)[1].strip().splitlines()[0].strip()


def parse_delimited_numeric_answer(text: str) -> int | None:
    """Parse only a numeric answer explicitly written after ``####``."""
    if "####" not in text:
        return None
    return parse_numeric_answer(f"#### {answer_after_delimiter(text)}")


def sample_solution(client: OpenAI, model: str, question: str, temperature: float,
                    max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        instructions=GENERATION_SYSTEM_PROMPT,
        input=question,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        store=False,
    )
    if not response.output_text.strip():
        raise RuntimeError("GPT-4 returned an empty solution")
    return response.output_text.strip()


def judge_solution(client: OpenAI, model: str, gold_answer: int,
                   solution: str, max_output_tokens: int) -> dict[str, Any]:
    """Label a solution, avoiding an LLM call for exact numeric matches."""
    candidate_answer = parse_delimited_numeric_answer(solution)
    if candidate_answer == gold_answer:
        return {"correct": True, "method": "exact_match"}

    candidate_text = answer_after_delimiter(solution) or "No answer after ####"
    prompt = (
        f"Reference numeric answer: {gold_answer}\n"
        f"Candidate final answer after ####: {candidate_text}"
    )
    schema = {
        "type": "json_schema",
        "name": "gsm8k_answer_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"correct": {"type": "boolean"}},
            "required": ["correct"],
            "additionalProperties": False,
        },
    }
    response = client.responses.create(
        model=model,
        instructions=JUDGE_SYSTEM_PROMPT,
        input=prompt,
        text={"format": schema},
        max_output_tokens=max_output_tokens,
        store=False,
    )
    try:
        judgment = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Judge returned non-JSON output: {response.output_text!r}") from exc
    if not isinstance(judgment.get("correct"), bool):
        raise RuntimeError(f"Judge returned an invalid judgment: {judgment!r}")
    judgment["method"] = "semantic_llm"
    return judgment


def existing_stats(output_path: Path) -> tuple[int, int, int, int]:
    """Return rows, objective-correct rows, judge-correct rows, agreements."""
    rows = objective_correct = judge_correct = agreements = 0
    if not output_path.exists():
        return rows, objective_correct, judge_correct, agreements
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            objective_correct += int(row.get("objective_correct", False))
            judge_correct += int(row.get("judge_correct", False))
            agreements += int(row.get("objective_correct") == row.get("judge_correct"))
    return rows, objective_correct, judge_correct, agreements


def print_stats(rows: int, objective_correct: int, judge_correct: int, agreements: int) -> None:
    if rows == 0:
        return
    print(
        f"Processed {rows} solutions | "
        f"solution accuracy: {objective_correct / rows:.3%} | "
        f"judge-positive rate: {judge_correct / rows:.3%} | "
        f"judge agreement: {agreements / rows:.3%}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--samples", type=int, default=1000, help="Number of GSM8K problems to use")
    parser.add_argument("--solutions-per-problem", type=int, default=2)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output", type=Path, default=Path("reward_models/data/gsm8k_gpt4.jsonl"))
    parser.add_argument("--model", default="gpt-4.1", help="GPT-4 model used for solution sampling")
    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="Cheaper model used only for semantic final-answer judging")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--print-every", type=int, default=10, help="Print metrics after this many new solutions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true", help="Append to an existing JSONL and include its metrics")
    args = parser.parse_args()

    load_dotenv()
    if args.samples < 1 or args.solutions_per_problem < 1:
        parser.error("--samples and --solutions-per-problem must be positive")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set")

    random.seed(args.seed)
    raw = load_dataset("openai/gsm8k", "main", split=args.split)
    raw = raw.shuffle(seed=args.seed).select(range(min(args.samples, len(raw))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        parser.error(f"Output already exists: {args.output}. Use --resume or choose another path.")
    mode = "a" if args.resume else "w"
    existing_by_problem: dict[int, list[dict]] = {}
    if args.resume:
        rows, objective_correct, judge_correct, agreements = existing_stats(args.output)
        with args.output.open(encoding="utf-8") as existing_handle:
            for line in existing_handle:
                if line.strip():
                    row = json.loads(line)
                    existing_by_problem.setdefault(row["problem_index"], []).append(row)
    else:
        rows, objective_correct, judge_correct, agreements = 0, 0, 0, 0
    client = OpenAI()

    print(f"Generating {len(raw) * args.solutions_per_problem} candidate solutions with {args.model}...", flush=True)
    with args.output.open(mode, encoding="utf-8") as handle:
        for problem_index, example in enumerate(raw):
            question = example["question"].strip()
            reference = example["answer"].strip()
            gold_answer = gold_numeric_answer(reference)
            existing_solutions = existing_by_problem.get(problem_index, [])
            if len(existing_solutions) >= args.solutions_per_problem:
                print(f"Skipping problem {problem_index} (already has {len(existing_solutions)} solutions)", flush=True)
                continue
            for sample_index in range(len(existing_solutions), args.solutions_per_problem):
                solution = sample_solution(client, args.model, question, args.temperature, args.max_output_tokens)
                
                candidate_answer = parse_delimited_numeric_answer(solution)
                objective_is_correct = candidate_answer == gold_answer
                judgment = judge_solution(client, args.judge_model, gold_answer, solution, args.max_output_tokens)
                row = {
                    "dataset": "openai/gsm8k",
                    "split": args.split,
                    "problem_index": problem_index,
                    "sample_index": sample_index,
                    "question": question,
                    "reference_solution": reference,
                    "gold_answer": gold_answer,
                    "solution": solution,
                    "candidate_answer": candidate_answer,
                    "objective_correct": objective_is_correct,
                    "judge_correct": judgment["correct"],
                    "label": int(judgment["correct"]),
                    "judge_method": judgment["method"],
                    "generation_model": args.model,
                    "judge_model": args.judge_model,
                    "temperature": args.temperature,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows += 1
                objective_correct += int(objective_is_correct)
                judge_correct += int(judgment["correct"])
                agreements += int(objective_is_correct == judgment["correct"])
                if rows % args.print_every == 0:
                    print_stats(rows, objective_correct, judge_correct, agreements)

    print(f"Saved dataset to {args.output}")
    print_stats(rows, objective_correct, judge_correct, agreements)


if __name__ == "__main__":
    main()
