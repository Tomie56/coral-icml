#!/usr/bin/env python3
"""
Minimal prompt-builder utilities expected by `mlp_steer_mmlu.py`.

The upstream repo referenced by this project had a separate "oracle" file
containing dataset-specific few-shot prompt templates. This repo version
imports it, but the file is missing. This module provides compatible
functions so `mlp_steer_mmlu.py` can run.

Design goals:
- Keep behavior deterministic and simple.
- Provide best-effort lm-eval-style prompts where possible.
- Support variable number of choices (2..8) across datasets.
"""

from __future__ import annotations

from typing import Dict, List, Optional


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _format_choices(question: str, choices: List[str], *, style: str) -> str:
    q = question.strip()
    lines: List[str] = [q]
    if style == "lmeval":
        # lm-eval harness commonly uses "A. <choice>"
        for L, c in zip(LETTERS, choices):
            lines.append(f"{L}. {str(c).strip()}")
    else:
        # legacy "A) <choice>"
        for L, c in zip(LETTERS, choices):
            lines.append(f"{L}) {str(c).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def get_generic_fewshot_examples(train_ds, num_fewshot: int) -> List[Dict]:
    """
    Generic helper used by `mlp_steer_mmlu.py` to sample few-shot examples.
    Expects a HuggingFace dataset object with __len__/__getitem__.
    We take the first N examples (matching "first_n" style).
    """
    n = min(int(num_fewshot), len(train_ds))
    out: List[Dict] = []
    for i in range(n):
        ex = train_ds[i]
        out.append(ex)
    return out


def _example_to_qa(ex: Dict, *, dataset: str) -> Dict[str, object]:
    """
    Normalize a training example dict to (question, choices, answer_idx).
    This is intentionally conservative; it covers the datasets used in the paper
    and in `mlp_steer_mmlu.py`.
    """
    if dataset == "arc":
        q = ex["question"]
        choices = ex["choices"]["text"]
        labels = ex["choices"].get("label", LETTERS[: len(choices)])
        gold = ex["answerKey"]
        answer_idx = labels.index(gold) if gold in labels else LETTERS.index(gold)
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "hellaswag":
        if "ctx_a" in ex and "ctx_b" in ex:
            q = f"{ex['ctx_a']} {ex['ctx_b']}".strip()
        else:
            q = ex.get("ctx", ex.get("context", ""))
        choices = list(ex["endings"])
        answer_idx = int(ex["label"])
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "siqa":
        q = f"{ex['context']} {ex['question']}".strip()
        choices = [ex["answerA"], ex["answerB"], ex["answerC"]]
        answer_idx = int(ex["label"]) - 1
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "openbookqa":
        q = ex["question_stem"]
        choices = ex["choices"]["text"]
        answer_idx = LETTERS.index(ex["answerKey"])
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "commonsenseqa":
        q = ex["question"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer_idx = labels.index(ex["answerKey"])
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "gsm_mc":
        q = ex["Question"]
        choices = [ex["A"], ex["B"], ex["C"], ex["D"]]
        answer_idx = LETTERS.index(ex["Answer"])
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "math_mc":
        q = ex["Question"]
        choices = [ex["A"], ex["B"], ex["C"], ex["D"]]
        answer_idx = LETTERS.index(ex["Answer"])
        return {"question": q, "choices": choices, "answer_idx": answer_idx}

    if dataset == "sciq":
        q = ex["question"]
        choices = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        answer_idx = 0
        return {"question": q, "choices": choices, "answer_idx": answer_idx, "support": ex.get("support", "")}

    if dataset == "race":
        # RACE includes an article/passage used as context.
        q = ex["question"]
        choices = ex["options"]
        answer_idx = LETTERS.index(ex["answer"])
        return {"question": q, "choices": choices, "answer_idx": answer_idx, "article": ex.get("article", "")}

    # Fallback: try common fields
    q = ex.get("question", ex.get("question_stem", ""))
    if "choices" in ex:
        if isinstance(ex["choices"], dict) and "text" in ex["choices"]:
            choices = ex["choices"]["text"]
        else:
            choices = ex["choices"]
    else:
        choices = [ex.get(k, "") for k in ["A", "B", "C", "D"] if k in ex]
    ans = ex.get("answerKey", ex.get("answer", "A"))
    answer_idx = LETTERS.index(ans) if ans in LETTERS else (int(ans) if str(ans).isdigit() else 0)
    return {"question": q, "choices": choices, "answer_idx": answer_idx}


def _build_fewshot_prompt(
    question: str,
    choices: List[str],
    fewshot_examples: List[Dict],
    *,
    style: str,
    dataset: str,
    include_context: Optional[Dict[str, str]] = None,
) -> str:
    parts: List[str] = []

    # Dataset-specific optional context prefix
    if include_context:
        if "article" in include_context and include_context["article"].strip():
            parts.append(include_context["article"].strip())
        if "support" in include_context and include_context["support"].strip():
            parts.append(include_context["support"].strip())

    # Few-shot examples
    for ex in fewshot_examples:
        norm = _example_to_qa(ex, dataset=dataset)
        ex_q = str(norm["question"])
        ex_choices = list(norm["choices"])
        ex_a = int(norm["answer_idx"])
        prompt = _format_choices(ex_q, ex_choices, style=style)
        prompt = prompt[:-len("Answer:")] + f"Answer: {LETTERS[ex_a]}"
        parts.append(prompt)

    # Test question (no answer)
    parts.append(_format_choices(question, choices, style=style))
    return "\n\n".join([p for p in parts if p is not None and str(p).strip() != ""])


# ---- Dataset-specific wrappers expected by mlp_steer_mmlu.py ----

def build_arc_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="legacy", dataset="arc")


def build_arc_fewshot_prompt_lmeval(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="arc")


def build_hellaswag_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="legacy", dataset="hellaswag")


def build_hellaswag_fewshot_prompt_lmeval(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="hellaswag")


def build_siqa_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="legacy", dataset="siqa")


def build_siqa_fewshot_prompt_lmeval(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="siqa")


def build_openbookqa_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="legacy", dataset="openbookqa")


def build_openbookqa_fewshot_prompt_lmeval(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="openbookqa")


def build_gsm_mc_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="legacy", dataset="gsm_mc")


def build_gsm_mc_fewshot_prompt_lmeval(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="gsm_mc")


def build_math_mc_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="legacy", dataset="math_mc")


def build_math_mc_fewshot_prompt_lmeval(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="math_mc")


def build_commonsenseqa_fewshot_prompt(question: str, choices: List[str], fewshot_examples: List[Dict]) -> str:
    # CommonsenseQA uses letter scoring commonly; keep choices visible.
    return _build_fewshot_prompt(question, choices, fewshot_examples, style="lmeval", dataset="commonsenseqa")


def build_sciq_fewshot_prompt_lmeval(
    question: str, choices: List[str], fewshot_examples: List[Dict], *, support: str = ""
) -> str:
    return _build_fewshot_prompt(
        question,
        choices,
        fewshot_examples,
        style="lmeval",
        dataset="sciq",
        include_context={"support": support or ""},
    )


def build_race_fewshot_prompt_lmeval(
    question: str, choices: List[str], fewshot_examples: List[Dict], *, article: str = ""
) -> str:
    return _build_fewshot_prompt(
        question,
        choices,
        fewshot_examples,
        style="lmeval",
        dataset="race",
        include_context={"article": article or ""},
    )

