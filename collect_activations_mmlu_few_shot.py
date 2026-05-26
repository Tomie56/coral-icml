#!/usr/bin/env python3
"""Collect activations from MMLU using lm-eval harness style few-shot prompting.

lm-eval MMLU format:
- Few-shot examples from 'dev' split (first_n sampler)
- Prompt: "{question}\nA. {choice0}\nB. {choice1}\nC. {choice2}\nD. {choice3}\nAnswer:"
- Answer: Just the letter (A, B, C, D)
- Default: 5-shot
"""
import argparse, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = ["A", "B", "C", "D"]


def build_mmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Build a single MMLU question prompt following lm-eval harness format.
    
    Format:
        {question}
        A. {choice0}
        B. {choice1}
        C. {choice2}
        D. {choice3}
        Answer:
    """
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_fewshot_prompt(
    question: str,
    choices: List[str],
    fewshot_examples: List[Dict],
) -> str:
    """
    Build few-shot prompt following lm-eval harness format for MMLU.
    
    Each few-shot example is formatted as:
        {question}
        A. {choice0}
        B. {choice1}
        C. {choice2}
        D. {choice3}
        Answer: {correct_letter}
    
    Examples are separated by double newlines.
    The test question ends with "Answer:" (no answer provided).
    """
    prompt_parts = []
    
    # Add few-shot examples
    for ex in fewshot_examples:
        ex_question = ex["question"]
        ex_choices = ex["choices"]
        ex_answer_idx = int(ex["answer"])
        correct_letter = LETTERS[ex_answer_idx]
        
        # Build the example prompt with answer
        ex_prompt = build_mmlu_prompt(ex_question, ex_choices)
        # Replace "Answer:" with "Answer: {letter}"
        ex_prompt = ex_prompt[:-len("Answer:")] + f"Answer: {correct_letter}"
        prompt_parts.append(ex_prompt)
    
    # Add test question (without answer)
    test_prompt = build_mmlu_prompt(question, choices)
    prompt_parts.append(test_prompt)
    
    return "\n\n".join(prompt_parts)


def get_fewshot_examples(dataset_name: str, subject: str, num_fewshot: int) -> List[Dict]:
    """
    Get few-shot examples from the dev split following lm-eval's first_n sampler.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "cais/mmlu")
        subject: MMLU subject name
        num_fewshot: Number of few-shot examples to retrieve
    
    Returns:
        List of example dicts with 'question', 'choices', 'answer' keys
    """
    if num_fewshot == 0:
        return []
    
    # Load dev split for few-shot examples
    dev_ds = load_dataset(dataset_name, subject, split="dev")
    
    # Take first N examples (lm-eval's first_n sampler)
    n_available = min(num_fewshot, len(dev_ds))
    examples = []
    for i in range(n_available):
        examples.append({
            "question": dev_ds[i]["question"],
            "choices": dev_ds[i]["choices"],
            "answer": dev_ds[i]["answer"],
        })
    
    return examples

@torch.no_grad()
def option_logprob_and_hidden(model, tok, prompt_ids, answer_ids, layers, pool):
    """
    Return:
      logprob (length-normalized) for P(answer | prompt),
      feats[layer] = pooled hidden vector (d,)
    """
    device = model.device
    input_ids = torch.tensor(prompt_ids + answer_ids, device=device).unsqueeze(0)
    attn_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
    )

    logits = out.logits[:, :-1, :]        # (1, T-1, V)
    labels = input_ids[:, 1:]             # (1, T-1)

    prompt_len = len(prompt_ids)
    start = prompt_len - 1
    end = labels.size(1)

    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    ans_logprobs = token_logprobs[:, start:end]
    logprob_len_norm = ans_logprobs.sum(dim=1) / (end - start)
    logprob = float(logprob_len_norm.item())

    feats = {}
    for li in layers:
        h = out.hidden_states[li].squeeze(0)  # (T, d)
        if pool == "answer_mean":
            pooled = h[prompt_len:, :].mean(dim=0)
        elif pool == "answer_final":
            pooled = h[prompt_len + (end - start) - 1, :]
        else:
            raise ValueError("pool must be {'answer_mean','answer_final'}")
        feats[li] = pooled.detach().float().cpu().numpy()

    return logprob, feats

def softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    z -= z.max()
    return np.exp(z) / np.exp(z).sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="cais/mmlu")    # MMLU on HF
    ap.add_argument("--subjects", type=str, default="all", help='Comma list or "all"')
    ap.add_argument("--split", default="test")
    ap.add_argument("--layers", type=str, default="all") # e.g. "10,20,28" or "all"
    ap.add_argument("--pool", choices=["answer_mean","answer_final"], default="answer_mean")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all per subject")
    ap.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N questions to reduce memory")
    ap.add_argument("--dtype", type=str, default="auto", help="Model dtype: auto, bfloat16, float16, float32")
    ap.add_argument("--out_dir", required=True)
    # Few-shot settings (lm-eval harness style)
    ap.add_argument("--num_fewshot", type=int, default=5,
                    help="Number of few-shot examples (default 5 for MMLU). Set to 0 for zero-shot.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    
    # Set dtype
    if args.dtype == "auto":
        dtype = "auto"
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    ).eval()
    print(f"Model loaded: {args.model_id} (dtype={model.dtype})")

    # Layers
    n_layers = model.config.num_hidden_layers
    if args.layers == "all":
        layers = list(range(n_layers + 1))  # 0..n_layers inclusive
    else:
        layers = [int(x) for x in args.layers.split(",")]
        for li in layers:
            assert 0 <= li <= n_layers, f"Layer {li} out of range 0..{n_layers}"

    # Subjects
    if args.subjects.strip().lower() == "all":
        subjects = get_dataset_config_names(args.dataset)
        # Filter out "auxiliary_train" AND "all" to avoid duplicates
        # "all" is a pre-concatenated version of individual subjects
        subjects = [s for s in subjects if s not in {"auxiliary_train", "all"}]
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    print(f"Subjects: {len(subjects)} → {subjects[:5]}{' ...' if len(subjects)>5 else ''}")
    print(f"Few-shot: {args.num_fewshot}-shot (from dev split, first_n sampler)")

    # Storage
    feats_layer = {li: [] for li in layers}
    p_hat_list, correct_list, miscal_mag_list, miscal_sign_list = [], [], [], []
    conf_list, pred_list, gold_list, ids = [], [], [], []
    option_idx_list, residual_prob_list, brier_score_list = [], [], []

    preds_path = out_dir / "preds.jsonl"
    with open(preds_path, "w") as f_pred:

        ex_counter = 0
        for subj in subjects:
            ds = load_dataset(args.dataset, subj, split=args.split)
            if args.max_examples and args.max_examples < len(ds):
                ds = ds.select(range(args.max_examples))
            
            # Get few-shot examples for this subject (from dev split)
            fewshot_examples = get_fewshot_examples(args.dataset, subj, args.num_fewshot)
            
            # Print first prompt for verification (only once per run)
            printed_first_prompt = (ex_counter > 0)

            for i, ex in enumerate(ds):
                # MMLU fields: "question" (str), "choices" (list of 4), "answer" (int index 0..3)
                q = ex["question"]
                choices = ex["choices"]
                gold_idx = int(ex["answer"])

                # Build few-shot prompt
                if args.num_fewshot > 0:
                    prompt = build_fewshot_prompt(q, choices, fewshot_examples)
                else:
                    prompt = build_mmlu_prompt(q, choices)
                
                # Print first prompt for verification
                if not printed_first_prompt:
                    print(f"\n[prompt] First prompt ({len(prompt)} chars):")
                    print("=" * 60)
                    if len(prompt) > 2000:
                        print(prompt[:1000] + "\n...[truncated]...\n" + prompt[-500:])
                    else:
                        print(prompt)
                    print("=" * 60 + "\n")
                    printed_first_prompt = True
                
                prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

                scores = []
                feats_acc = {li: [] for li in layers}

                # Score each answer choice (just the letter, following lm-eval)
                for opt_idx, letter in enumerate(LETTERS[:len(choices)]):
                    # lm-eval uses just the letter as the answer
                    ans_text = f" {letter}"
                    ans_ids = tok(ans_text, add_special_tokens=False)["input_ids"]
                    lp, feats = option_logprob_and_hidden(model, tok, prompt_ids, ans_ids, layers, pool=args.pool)
                    scores.append(lp)
                    for li in layers:
                        feats_acc[li].append(feats[li])

                probs = softmax_np(scores)  # (4,)
                pred = int(probs.argmax())
                conf = float(probs[pred])
                p_correct = float(probs[gold_idx])  # probability assigned to correct answer

                # Collect activations for ALL 4 options (one row per option)
                for opt_idx in range(len(choices)):
                    is_correct = (opt_idx == gold_idx)
                    p_option = float(probs[opt_idx])
                    
                    # Calculate labels
                    y = 1 if opt_idx == pred else 0  # is this the predicted option?
                    is_gold = 1 if is_correct else 0
                    
                    # Residual probability
                    if is_correct:
                        residual_prob = 1.0 - p_correct
                    else:
                        residual_prob = 0.0 - p_option  # negative value
                    
                    # Brier score component
                    if is_correct:
                        brier = (1.0 - p_correct) ** 2
                    else:
                        brier = (0.0 - p_option) ** 2  # = p_option^2
                    
                    # Miscalibration metrics (based on if model picked this option)
                    if opt_idx == pred:
                        p_hat = conf
                        miscal_mag = abs(p_hat - is_gold)
                        miscal_sign = 1 if p_hat > is_gold else (-1 if p_hat < is_gold else 0)
                    else:
                        p_hat = p_option
                        miscal_mag = 0.0  # not the predicted option
                        miscal_sign = 0
                    
                    # Store activations for this option
                    for li in layers:
                        feats_layer[li].append(feats_acc[li][opt_idx])
                    
                    # Store labels
                    ids.append(f"mmlu_{subj}_{args.split}_{i:05d}_opt{opt_idx}")
                    option_idx_list.append(opt_idx)
                    p_hat_list.append(p_hat)
                    correct_list.append(is_gold)
                    miscal_mag_list.append(miscal_mag)
                    miscal_sign_list.append(miscal_sign)
                    conf_list.append(p_option)  # probability of this specific option
                    pred_list.append(pred)  # what the model predicted overall
                    gold_list.append(gold_idx)  # what the correct answer is
                    residual_prob_list.append(residual_prob)
                    brier_score_list.append(brier)

                # Write prediction summary (one per question, unchanged format)
                item = {
                    "id": f"mmlu_{subj}_{args.split}_{i:05d}",
                    "probs": [float(x) for x in probs],
                    "gold": gold_idx,
                    "pred": pred,
                    "conf": conf,
                    "subject": subj
                }
                f_pred.write(json.dumps(item) + "\n")

                ex_counter += 1
                if ex_counter % 50 == 0:
                    print(f"[{ex_counter}] {subj}: processed {i+1}/{len(ds)}")
                
                # Periodic checkpoint to reduce memory
                if ex_counter % args.save_every == 0:
                    print(f"[checkpoint] Saving at {ex_counter} questions...")
                    torch.cuda.empty_cache()  # Clear CUDA cache

    # Stack features
    feat_np = {f"layer_{li}": np.stack(feats_layer[li], axis=0) for li in layers}
    labels = {
        "ids": np.array(ids),
        "option_idx": np.array(option_idx_list, dtype=np.int32),
        "p_hat": np.array(p_hat_list, dtype=np.float32),
        "correct": np.array(correct_list, dtype=np.int32),
        "miscal_mag": np.array(miscal_mag_list, dtype=np.float32),
        "miscal_sign": np.array(miscal_sign_list, dtype=np.int8),
        "conf": np.array(conf_list, dtype=np.float32),
        "pred": np.array(pred_list, dtype=np.int32),
        "gold": np.array(gold_list, dtype=np.int32),
        "residual_prob": np.array(residual_prob_list, dtype=np.float32),
        "brier_score": np.array(brier_score_list, dtype=np.float32),
        "layers": np.array(layers, dtype=np.int32),
    }

    np.savez_compressed(out_dir / "probe_data.npz", **feat_np, **labels)

    print("\nWrote:")
    print(f"  predictions: {preds_path}")
    print(f"  probe data : {out_dir/'probe_data.npz'}")
    print(f"\nFew-shot setting: {args.num_fewshot}-shot (lm-eval harness style)")

if __name__ == "__main__":
    main()
