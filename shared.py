from __future__ import annotations

import sys
sys.modules["tensorflow"] = None

import ast
import json
import logging
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy as sp_entropy

logger = logging.getLogger(__name__)


TASK_PROMPTS = {
    "offensive_en": {
        "system_msg": (
            "You are an AI expert in text classification. "
            "You will read a text. Does the text contain offensive language? "
            "Answer only with 0 for NOT and 1 for YES. "
            "Do not write an introduction or summary. Classify always the text."
        ),
        "zero_shot_user": (
            "You are an AI expert in text classification. "
            "You will read a text. Does the text contain offensive language? "
            "Now classify the following text: {text}\n"
            "{{'text': '{text_short}', 'label': }}"
        ),
        "examples_header": "\nHere are some examples:",
        "classify_user": (
            "Now classify the following text: {text}\n"
            "Answer only with 0 or 1."
        ),
        "auto_cot_system": "You are an AI expert in text classification.",
        "auto_cot_user": (
            "Text: {text}\n"
            "You will read a text. Does the text contain offensive language? "
            "Think step by step and present your reasoning in one sentence. "
            "then you must answer 0 (no) or 1 (yes)."
        ),
        "cot_split_word": "Therefore",
        "cot_conclusion": "Therefore the answer is {label}",
    
        "positive_keyword": "offensive",
        "negative_keyword": "not offensive",
    },
    "irony_it": {
        "system_msg": (
            "Sei un esperto di IA specializzato in classificazione di testi. "
            "Leggerai un testo. Il testo contiene linguaggio ironico? "
            "Rispondi solo con 0 per NO e 1 per SÌ. "
            "Non scrivere un'introduzione o una conclusione. Classifica sempre il testo."
        ),
        "zero_shot_user": (
            "Sei un esperto di classificazione del testo. "
            "Leggerai un testo. Il testo contiene linguaggio ironico? "
            "Classifica il seguente testo: {text}\n"
            "{{'text': '{text_short}', 'label': }}"
        ),
        "examples_header": "\nEcco alcuni esempi:",
        "classify_user": (
            "Ora classifica il seguente testo: {text}\n"
            "Rispondi solo con 0 o 1."
        ),
        "auto_cot_system": "Sei un esperto di IA specializzato in classificazione di testi.",
        "auto_cot_user": (
            "Text: {text}\n"
            "Leggerai un testo. Il testo contiene linguaggio ironico? "
            "Pensa passo dopo passo e presenta il tuo ragionamento in una frase. "
            "poi devi rispondere 0 (no) o 1 (sì)."
        ),
        "cot_split_word": "Quindi",
        "cot_conclusion": "Quindi la risposta è {label}",
        
        "positive_keyword": "ironico",
        "negative_keyword": "non ironico",
    },
}


_current_task = "offensive_en"


def set_task(task: str):
  
    global _current_task
    if task not in TASK_PROMPTS:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(TASK_PROMPTS.keys())}")
    _current_task = task
    logger.info("Task set to: %s", task)


def get_task_config(task: str = None) -> dict:

    t = task or _current_task
    return TASK_PROMPTS[t]



def parse_annotations(val):
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            return [int(x) for x in ast.literal_eval(val)]
        except (ValueError, SyntaxError):
            return [int(x.strip()) for x in val.strip("[]").split(",") if x.strip()]
    return [int(val)]


def load_dataset(path, max_instances=None):
    df = pd.read_csv(path)
    if max_instances:
        df = df.head(max_instances)
    logger.info("Loaded %d instances from %s", len(df), path)
    return df



class SLM:


    def __init__(self, model_name, device="auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading %s ...", model_name)
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"device_map": device}
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            print(">>> 4-bit quantization ENABLED")
        except ImportError:
            kwargs["torch_dtype"] = torch.float16
            print(">>> NO quantization, using float16")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        mem = self.model.get_memory_footprint() / 1e9
        print(f">>> Model memory: {mem:.1f} GB")

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            self.tok.pad_token_id = self.tok.eos_token_id

        logger.info("Model on %s", self.device)

    def classify(self, messages):
 
        torch.cuda.empty_cache()
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=5,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def generate(self, messages, max_new=80):

        torch.cuda.empty_cache()
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new,
                do_sample=True, temperature=0.95,
                pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def embed_layer(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "transformer"):
            return self.model.transformer.wte
        raise AttributeError(f"Cannot find embedding layer for {self.name}")



def build_zero_shot_messages(text, task=None):
    """Zero-shot: system + classify this text."""
    cfg = get_task_config(task)
    return [
        {"role": "system", "content": cfg["system_msg"]},
        {"role": "user", "content": cfg["zero_shot_user"].format(
            text=text, text_short=text[:50])},
    ]


def build_io_messages(text, shots, task=None):
    """
    IO baseline: system + shot examples + test text.
    shots = [{"text": ..., "label": 0 or 1}, ...]
    """
    cfg = get_task_config(task)
    messages = [{"role": "system", "content": cfg["system_msg"] + cfg["examples_header"]}]

    for shot in shots:
        messages.append({"role": "user", "content": shot["text"]})
        messages.append({"role": "assistant", "content": str(shot["label"])})

    messages.append({"role": "user", "content": cfg["classify_user"].format(text=text)})
    return messages


def build_rationale_messages(text, shots, task=None):
    """
    Auto-CoT / Self-AMPLIFY style: system + shots with rationale + test text.
    shots = [{"text": ..., "label": 0 or 1, "rationale": "..."}, ...]
    """
    cfg = get_task_config(task)
    messages = [{"role": "system", "content": cfg["system_msg"] + cfg["examples_header"]}]

    for shot in shots:
        messages.append({"role": "user", "content": shot["text"]})
        response = ""
        if shot.get("rationale"):
            response += shot["rationale"] + "\n"
        response += str(shot["label"])
        messages.append({"role": "assistant", "content": response})

    messages.append({"role": "user", "content": cfg["classify_user"].format(text=text)})
    return messages



def extract_label(generated_text, task=None):
   
    cfg = get_task_config(task)
    text = generated_text.strip()

  
    if text in ("0", "1"):
        return int(text)

    if text and text[0] in ("0", "1"):
        return int(text[0])

    m = re.search(r'\b([01])\b', text)
    if m:
        return int(m.group(1))

    lo = text.lower()
    if cfg["negative_keyword"] in lo:
        return 0
    if cfg["positive_keyword"] in lo:
        return 1
    return -1


def predict_dataset(slm, df_test, message_fn, condition_name, output_csv, task=None):
    cfg = get_task_config(task)
    rows = []
    for i, (idx, row) in enumerate(df_test.iterrows()):
        text = str(row["text"])
        messages = message_fn(text)
        output = slm.classify(messages)
        pred = extract_label(output, task=task)
        rows.append({
            "idx": int(idx),
            "text": text,
            "hard_label": int(row["hard_label"]),
            "predicted_label": pred,
            "raw_output": output[:300],
            "condition": condition_name,
            "mace_entropy": row.get("mace_entropy", None),
            "disagreement": row.get("disagreement", None),
        })
        if (i + 1) % 20 == 0:
            logger.info("    [%s] %d/%d predicted", condition_name, i + 1, len(df_test))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    logger.info("    Saved %d predictions → %s", len(df_out), output_csv)
    return df_out