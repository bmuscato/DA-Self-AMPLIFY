from __future__ import annotations


import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import LayerDeepLift

from shared import SLM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


STOP_WORDS = {
   
    "-", ".", ",", ";", "!", "?", "'", ":", "\u2019", "___", "_",
    "(A)", "(B)", "(C)", "(D)", "(E)", "(F)",
    "(a)", "(b)", "(c)", "(d)", "(e)", "(f)",

   
    "the", "a", "an", "to", "is", "of", "on", "in", "are", "and",
    "does", "was", "were", "be", "been", "has", "have", "had",
    "do", "did", "for", "with", "at", "by", "or", "but", "not",
    "that", "this", "it", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "us", "them",
    "am", "will", "would", "could", "should", "can", "may",
    "if", "so", "as", "than", "then", "just", "also", "very",
    "no", "yes", "all", "some", "any", "each", "every",
    "from", "about", "into", "up", "out", "down", "over",
    "what", "which", "who", "how", "when", "where", "why",
    "there", "here", "been", "being",

   
    "il", "lo", "la", "i", "gli", "le", "l",
    "un", "uno", "una", "un\u2019",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "e", "ed", "o", "ma", "n\u00e9", "che", "se", "anche",
    "per\u00f2", "quindi", "oppure", "perch\u00e9",
    "mi", "ti", "ci", "vi", "si", "lo", "la", "li", "le", "ne",
    "io", "tu", "lui", "lei", "noi", "voi", "loro",
    "me", "te", "s\u00e9",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "chi", "cui", "quale", "quali",
    "\u00e8", "sono", "sei", "siamo", "siete",
    "era", "ero", "eravamo", "erano",
    "stato", "stata", "stati", "state",
    "ho", "ha", "hai", "abbiamo", "avete", "hanno",
    "aveva", "avevo", "avevamo", "avevano",
    "avuto",
    "fare", "fa", "fatto",
    "essere", "avere",
    "pu\u00f2", "potere", "deve", "dovere", "vuole", "volere",
    "non", "pi\u00f9", "molto", "tanto", "poco", "gi\u00e0",
    "ancora", "sempre", "mai", "qui", "l\u00e0",
    "come", "dove", "quando", "cosa",
    "tutto", "tutti", "tutta", "tutte",
    "altro", "altri", "altra", "altre",
    "proprio", "stesso", "stessa",
    "cos\u00ec", "solo", "ancora", "ogni",
}



def select_random_shots(df, n_per_class=10, seed=42, shuffle=False):
   
    rng = np.random.default_rng(seed)
    indices = []
    for cls in sorted(df["hard_label"].unique()):  # 0 first, then 1
        pool = df[df["hard_label"] == cls].index.tolist()
        n = min(n_per_class, len(pool))
        selected = list(rng.choice(pool, size=n, replace=False))
        indices.extend(selected)

    if shuffle:
        rng.shuffle(indices)

    logger.info("  Selected %d random shots (%d/class, shuffle=%s)",
                len(indices), n_per_class, shuffle)
    return indices



def _find_subsequence(full_ids: list[int], sub_ids: list[int]) -> int | None:
    """Find the start index of sub_ids within full_ids."""
    for i in range(len(full_ids) - len(sub_ids) + 1):
        if full_ids[i:i + len(sub_ids)] == sub_ids:
            return i
    return None


def _aggregate_subtoken_attribution(
    tokenizer, token_ids: list[int], raw_attr: np.ndarray
) -> tuple[list[str], list[float]]:
    """
    Aggregate sub-token attributions back to word-level,
    zeroing out stop words.
    """
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    words_clean = full_text.split()

    subtokens = [
        tokenizer.decode([tid]).replace(" ", "").replace("\u2581", "")
        for tid in token_ids
    ]

    word_attrs = []
    k = 0
    query = ""
    attr_acc = 0.0

    for i, st in enumerate(subtokens):
        if k >= len(words_clean):
            break
        target_word = words_clean[k].replace("\u2581", "")
        query += st
        attr_acc += raw_attr[i]

        if query == target_word:
            if query.lower() in STOP_WORDS:
                attr_acc = 0.0
            word_attrs.append(attr_acc)
            k += 1
            query = ""
            attr_acc = 0.0

    while len(word_attrs) < len(words_clean):
        word_attrs.append(0.0)

    return words_clean, word_attrs


#deeplift rationale generation 

def generate_deeplift(slm, text: str, label_str: str, top_k: int = 4,
                      max_new_tokens: int = 1) -> str:
    
    messages = [
        {"role": "user", "content": text},
    ]
    prompt = slm.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += "The answer is ("

    input_ids = slm.tok(prompt, return_tensors="pt")["input_ids"].to(slm.device)

    text_tok_ids = slm.tok.encode(text, add_special_tokens=False)
    full_tok_ids = input_ids[0].tolist()

    start = _find_subsequence(full_tok_ids, text_tok_ids)
    if start is None:
        logger.warning("  Could not locate text tokens exactly, using heuristic")
        start = 0
    end = start + len(text_tok_ids)

    pad_id = slm.tok.pad_token_id
    if pad_id is None:
        pad_id = slm.tok.eos_token_id or 0

    embed_layer = slm.embed_layer()

    class _Wrapper(torch.nn.Module):
        def __init__(self, model, embed):
            super().__init__()
            self.model = model
            self.embed = embed

        def forward(self, input_ids):
            emb = self.embed(input_ids)
            logits = self.model(inputs_embeds=emb).logits[:, -1, :]
            return logits

    wrapper = _Wrapper(slm.model, embed_layer)
    ldi = LayerDeepLift(wrapper, embed_layer)

    target_token_ids = slm.tok.encode(label_str, add_special_tokens=False)
    if not target_token_ids:
        target_token_ids = [slm.tok.encode(label_str)[0]]
    n_generate = max(max_new_tokens, len(target_token_ids))

    attribution_list = []
    idx = input_ids.clone()

    for step in range(n_generate):
        if step < len(target_token_ids):
            idx_next = target_token_ids[step]
        else:
            with torch.no_grad():
                output = slm.model(idx)
                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.argmax(probs).item()

        baseline = idx.clone()
        baseline[0, start:end] = pad_id

        attr = ldi.attribute(
            idx,
            baselines=baseline,
            target=idx_next,
            return_convergence_delta=False,
        )
        attr_step = attr.sum(dim=-1)[0, start:end]
        attribution_list.append(attr_step.detach().cpu())

        if idx_next == slm.tok.eos_token_id:
            break

    final_attributions = torch.mean(torch.stack(attribution_list), dim=0)
    attr_sum = final_attributions.sum()
    if attr_sum != 0:
        final_attributions = final_attributions / attr_sum

    attr_scores = final_attributions.numpy()

    text_token_ids_slice = full_tok_ids[start:end]
    words, word_attrs = _aggregate_subtoken_attribution(
        slm.tok, text_token_ids_slice, attr_scores
    )

    word_attrs_np = np.array(word_attrs)
    k = min(top_k, len(words))
    top_indices = np.argsort(np.abs(word_attrs_np))[-k:]
    top_indices = np.sort(top_indices)

    top_words = []
    for idx in top_indices:
        w = words[idx].strip(".,;:!?\"'()[]")
        if w and w not in top_words:
            top_words.append(w)

    if not top_words:
        top_words = ["text"]

    return _format_keyword_rationale(top_words, label_str)


def _format_keyword_rationale(words: list[str], label_str: str) -> str:
    words = [w.replace("<BOS_TOKEN>", "") for w in words]
    n = len(words)
    if n == 1:
        kw_str = f"'{words[0]}'"
    else:
        kw_str = ", ".join(f"'{w}'" for w in words[:-1])
        kw_str += f", and '{words[-1]}'"
    return (
        f"The {n} keywords {kw_str} are important "
        f"to predict that the answer is {label_str}"
    )


def deeplift_fallback(text: str, label_str: str, top_k: int = 4) -> str:
    
    ws = [
        w for w in re.findall(r'\b[a-zA-Z\u00C0-\u024F]+\b', text.lower())
        if w not in STOP_WORDS and len(w) > 2
    ][:top_k]
    if not ws:
        ws = ["text"]
    return _format_keyword_rationale(ws, label_str)


#ph-cot rationale generation 

def generate_phcot(slm, text: str, label_str: str, n_steps: int = 3) -> str:

    step_template = ", ".join(
        f"step{i+1}" if i < n_steps - 1 else f"and step{i+1}"
        for i in range(n_steps)
    )

    user_content = (
        f"generate an explanation with only one sentence per step\n"
        f"Example: The answer is {label_str}, {n_steps}-step explanation: "
        f"{step_template}\n\n"
        f"{text}"
    )

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"The answer is {label_str}, {n_steps}-step explanation: step1:"
    )

    inputs = slm.tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(slm.device)
    attention_mask = inputs["attention_mask"].to(slm.device)

    len_input = input_ids.shape[1]
    with torch.no_grad():
        outputs = slm.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=slm.tok.pad_token_id,
            max_new_tokens=300,
            min_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
        )
    explanation = slm.tok.decode(
        outputs[0][len_input:], skip_special_tokens=True
    ).strip()

    full_explanation = "step1:" + explanation if explanation else "step1: The text contains relevant indicators."

    return (
        f"{n_steps}-step rationale: {full_explanation}, "
        f"therefore the answer is {label_str}"
    )



def run_step3(data_path, model_name, top_k=4, n_steps=3,
              device="auto", output=None,
              shot_strategy="all", n_per_class=10, seed=42):
    name = Path(data_path).stem
    logger.info("=" * 64)
    logger.info("  STEP 3 — Rationale generation: %s", name)
    logger.info("  DeepLift top_k=%d | Ph-CoT n_steps=%d", top_k, n_steps)
    logger.info("  Shot strategy: %s", shot_strategy)
    logger.info("=" * 64)

    df = pd.read_csv(data_path)

  
    if shot_strategy == "all":
        indices_to_explain = sorted(df.index.tolist())
    elif shot_strategy == "random_ordered":
        indices_to_explain = select_random_shots(df, n_per_class, seed, shuffle=False)
    elif shot_strategy == "random_shuffled":
        indices_to_explain = select_random_shots(df, n_per_class, seed, shuffle=True)
    else:
        raise ValueError(f"Unknown shot_strategy: {shot_strategy}")

    logger.info("  %d instances to explain", len(indices_to_explain))

    slm = SLM(model_name, device)
    cache = {}

    for i, idx in enumerate(indices_to_explain):
        row = df.loc[idx]
        text = str(row["text"])
        label = str(int(row["hard_label"]))

        logger.info(
            "\n  [%d/%d] idx=%d  label=%s", i + 1, len(indices_to_explain), idx, label
        )

       #deeplift top_k
        try:
            dl = generate_deeplift(slm, text, label, top_k=top_k)
        except Exception as e:
            logger.warning("    DeepLift failed (%s), using fallback", e)
            dl = deeplift_fallback(text, label, top_k=top_k)
        logger.info("    DL: %s", dl[:120])

        #ph-cot n_steps
        pc = generate_phcot(slm, text, label, n_steps=n_steps)
        logger.info("    PC: %s", pc[:120])

        cache[str(idx)] = {"deeplift": dl, "phcot": pc}

    strategy_suffix = f"_{shot_strategy}" if shot_strategy != "all" else ""
    out = output or f"step3_{name}_{model_name.split('/')[-1]}{strategy_suffix}_rationales.json"
    with open(out, "w") as f:
        json.dump(
            {
                "dataset": name,
                "model": model_name,
                "top_k": top_k,
                "n_steps": n_steps,
                "shot_strategy": shot_strategy,
                "n_per_class": n_per_class if shot_strategy != "all" else None,
                "seed": seed if shot_strategy != "all" else None,
                #"indices": indices_to_explain,
                "indices": [int(i) for i in indices_to_explain],
                "rationales": cache,
            },
            f,
            indent=2,
        )
    logger.info("\n  Saved to %s", out)
    logger.info("=" * 64)
    return cache


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 3: Rationale generation")
    p.add_argument("--train", required=True, help="CSV with 'text' and 'hard_label' columns")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    p.add_argument("--shot_strategy", default="all",
                    choices=["all", "random_ordered", "random_shuffled"],
                    help="'all' explains every row; 'random_ordered' picks "
                         "n_per_class random samples per class (neg first); "
                         "'random_shuffled' picks n_per_class random (shuffled)")
    p.add_argument("--n_per_class", type=int, default=10,
                    help="Samples per class for random strategies (default: 10)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_step3(
        args.train, args.model,
        args.top_k, args.n_steps, args.device, args.output,
        args.shot_strategy, args.n_per_class, args.seed,
    )