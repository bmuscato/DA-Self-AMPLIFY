from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch

from shared import SLM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)




TASK_CONFIGS = {
    "offensive_en": {
        "system_msg": "You are an AI expert in text classification.",
        "task_question": "Does the text contain offensive language?",
        "answer_instruction": "Then you must answer 0 (no) or 1 (yes).",
        "kw_preprompt": (
            "You are going to read a text {task_question} "
            "Generate {top_k} keywords providing hints and generate the right single answer 0 (no) or 1 (yes).\n"
            "Output example: The {top_k} keywords {kw_example} "
            "are important to predict that the answer is 0"
        ),
        "phcot_preprompt": (
            "You are going to read a text {task_question} "
            "Generate a concise {n_steps}-step explanation, with only one sentence per step, "
            "and generate the right single answer 0 (no) or 1 (yes).\n"
            "Output example: {n_steps}-step rationale: {step_example} "
            "Therefore the answer is 0"
        ),
        "deeplift_prefill": "The {top_k} keywords '",
        "phcot_prefill": "{n_steps}-step rationale: ",
    },
    "irony_it": {
        "system_msg": "Sei un esperto di IA specializzato in classificazione di testi.",
        "task_question": "Il testo contiene linguaggio ironico?",
        "answer_instruction": "Poi devi rispondere 0 (no) o 1 (sì).",
        "kw_preprompt": (
            "Leggerai un testo. {task_question} "
            "Genera {top_k} parole chiave che forniscono indizi e genera la risposta corretta 0 (no) o 1 (sì).\n"
            "Esempio di output: Le {top_k} parole chiave {kw_example} "
            "sono importanti per predire che la risposta è 0"
        ),
        "phcot_preprompt": (
            "Leggerai un testo. {task_question} "
            "Genera una spiegazione concisa di {n_steps} passaggi, con una sola frase per passaggio, "
            "e genera la risposta corretta 0 (no) o 1 (sì).\n"
            "Esempio di output: Ragionamento in {n_steps} passaggi: {step_example} "
            "Quindi la risposta è 0"
        ),
        "deeplift_prefill": "Le {top_k} parole chiave '",
        "phcot_prefill": "Ragionamento in {n_steps} passaggi: ",
    },
}


#preprompt 

def build_preprompt(cfg: dict, explainer: str, top_k: int = 4, n_steps: int = 3) -> str:
    if explainer == "deeplift":
        kw_parts = [
            f"and 'word{i+1}'" if i == top_k - 1 else f"'word{i+1}'"
            for i in range(top_k)
        ]
        return cfg["kw_preprompt"].format(
            task_question=cfg["task_question"],
            top_k=top_k,
            kw_example=", ".join(kw_parts),
        )
    elif explainer == "phcot":
        step_example = " ".join(f"Step{i+1}." for i in range(n_steps))
        return cfg["phcot_preprompt"].format(
            task_question=cfg["task_question"],
            n_steps=n_steps,
            step_example=step_example,
        )
    else:
        raise ValueError(f"Unknown explainer: {explainer}")


#llama prompt builder

def build_llama3_prompt(messages: list[dict], assistant_prefill: str = "") -> str:
    
    prompt = "<|begin_of_text|>"

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        prompt += content

       
        is_last = (i == len(messages) - 1)
        if is_last and role == "assistant":
            pass  
        else:
            prompt += "<|eot_id|>"

 
    if messages[-1]["role"] != "assistant":
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt += assistant_prefill

    return prompt


def select_stratified_shots(
    df_train: pd.DataFrame,
    shot_indices: list[int],
    label_space: list[str],
    shots_per_class: int,
) -> list[int]:
    
    per_class = {}
    for label in label_space:
        class_indices = [
            idx for idx in shot_indices
            if str(int(df_train.loc[idx, "hard_label"])) == label
        ]
        per_class[label] = class_indices[:shots_per_class]
        logger.info("    Class %s: %d candidates → selected %d",
                     label, len([
                         idx for idx in shot_indices
                         if str(int(df_train.loc[idx, "hard_label"])) == label
                     ]), len(per_class[label]))

   
    selected = []
    max_len = max(len(v) for v in per_class.values())
    for i in range(max_len):
        for label in label_space:
            if i < len(per_class[label]):
                selected.append(per_class[label][i])

    return selected



def build_icl_messages(
    df_train: pd.DataFrame,
    shot_indices: list[int],
    rationales: dict,
    explainer: str,
    cfg: dict,
    top_k: int = 4,
    n_steps: int = 3,
) -> list[dict]:
   
    preprompt = build_preprompt(cfg, explainer, top_k, n_steps)

    messages = [{"role": "system", "content": cfg["system_msg"]}]

    for i, idx in enumerate(shot_indices):
        row = df_train.loc[idx]
        text = str(row["text"])

        
        rat = rationales[str(idx)][explainer]

       
        if i == 0:
            user_content = f"Text: {text}\n{preprompt}"
        else:
            user_content = (
                f"Text: {text}\n"
                f"{cfg['task_question']} "
                f"{cfg['answer_instruction']}"
            )

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": rat})

    return messages




def extract_answer(text: str, label_space: list[str]) -> str | None:
    
    text_lower = text.lower()

    for label in label_space:
        patterns = [
            # English
            f"the answer is {label}",
            f"answer is {label}",
            f"predict that the answer is {label}",
            f"therefore the answer is {label}",
            # Italian
            f"la risposta è {label}",
            f"risposta è {label}",
            f"quindi la risposta è {label}",
            f"predire che la risposta è {label}",
        ]
        for pat in patterns:
            if pat in text_lower:
                return label

 
    matches = list(re.finditer(r'\b([01])\b', text))
    if matches:
        return matches[-1].group(1)

    return None



def run_step4(
    train_path: str,
    test_path: str,
    rationales_path: str,
    model_name: str,
    explainer: str = "deeplift",
    task: str = "offensive_en",
    top_k: int = 4,
    n_steps: int = 3,
    max_new_tokens: int = 150,
    shots_per_class: int = 10,
    device: str = "auto",
    output: str = None,
):
    name = Path(test_path).stem
    cfg = TASK_CONFIGS[task]

    logger.info("=" * 64)
    logger.info("  STEP 4 — ICL Inference (rationales from step 3)")
    logger.info("  Task: %s | Explainer: %s | Test: %s", task, explainer, name)
    logger.info("  Shots per class: %d", shots_per_class)
    logger.info("=" * 64)

   
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    with open(rationales_path) as f:
        rationale_data = json.load(f)
    rationales = rationale_data["rationales"]


    all_labels = sorted(df_train["hard_label"].unique())
    label_space = [str(int(l)) for l in all_labels]
    logger.info("  Label space: %s", label_space)


    if "indices" in rationale_data:
        shot_indices = rationale_data["indices"]
        logger.info("  Available indices from rationales JSON: %d", len(shot_indices))
    else:
        shot_indices = df_train.index.tolist()
    logger.info("  Available train samples: %d", len(shot_indices))


    shot_indices = select_stratified_shots(
        df_train, shot_indices, label_space, shots_per_class
    )
    logger.info("  Selected %d shots total (%d per class, interleaved)",
                len(shot_indices), shots_per_class)

   
    missing = [str(idx) for idx in shot_indices if str(idx) not in rationales]
    if missing:
        logger.error("  Missing rationales for %d indices (first 5: %s)",
                      len(missing), missing[:5])
        raise KeyError(
            f"Rationale JSON is missing {len(missing)} entries. "
            f"Re-run step 3 or check your train CSV."
        )


    slm = SLM(model_name, device)


    icl_messages = build_icl_messages(
        df_train, shot_indices, rationales,
        explainer, cfg, top_k, n_steps
    )

    logger.info("\n  --- ICL Prompt Preview (%d messages, %d shots) ---",
                len(icl_messages), len(shot_indices))
    for msg in icl_messages[:6]:
        role = msg["role"].upper()
        content = msg["content"][:120]
        logger.info("  [%s] %s...", role, content)
    logger.info("  --- End Preview ---\n")

    # max_new_tokens per explainer 
    # DeepLift: short output (~30 tokens for keywords + answer)
    # PhCoT: longer output (~200 tokens for 3-step reasoning + answer)
    if explainer == "deeplift":
        max_new_tokens = min(max_new_tokens, 80)
    elif explainer == "phcot":
        max_new_tokens = max(max_new_tokens, 300)

    logger.info("  Decoding: greedy (do_sample=False) | max_new_tokens=%d", max_new_tokens)

   
    if explainer == "deeplift":
        assistant_prefill = cfg["deeplift_prefill"].format(top_k=top_k)
    elif explainer == "phcot":
        assistant_prefill = cfg["phcot_prefill"].format(n_steps=n_steps)
    else:
        assistant_prefill = ""

    #inference loop 
    results = []
    for i, idx in enumerate(df_test.index):
        row = df_test.loc[idx]
        text = str(row["text"])
        true_label = str(int(row["hard_label"]))

     
        torch.cuda.empty_cache()

    
        test_messages = icl_messages.copy()
        test_messages.append({
            "role": "user",
            "content": (
                f"Text: {text}\n"
                f"{cfg['task_question']} "
                f"{cfg['answer_instruction']}"
            )
        })

 
        prompt = build_llama3_prompt(test_messages, assistant_prefill)

      
        inputs = slm.tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(slm.device)
        attention_mask = inputs["attention_mask"].to(slm.device)

        len_input = input_ids.shape[1]


        with torch.no_grad():
            outputs = slm.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=slm.tok.pad_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = slm.tok.decode(
            outputs[0][len_input:], skip_special_tokens=True
        ).strip()


        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()


        full_answer = assistant_prefill + generated


        pred_label = extract_answer(full_answer, label_space)

        if (i + 1) % 10 == 0 or i < 3:
            logger.info(
                "  [%d/%d] true=%s pred=%s | %s",
                i + 1, len(df_test), true_label, pred_label,
                full_answer[:120]
            )

        results.append({
            "idx": int(idx),
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "full_answer": full_answer,
        })


    results_df = pd.DataFrame(results)
    valid = results_df[results_df["pred_label"].notna()]
    accuracy = (valid["true_label"] == valid["pred_label"]).mean()
    unparsed = results_df["pred_label"].isna().sum()

    logger.info("\n" + "=" * 64)
    logger.info("  RESULTS (%s, %s, %s)", task, explainer, name)
    logger.info("  Shots: %d (%d per class)", len(shot_indices), shots_per_class)
    logger.info("  Accuracy:  %.1f%% (%d/%d)", accuracy * 100,
                int((valid["true_label"] == valid["pred_label"]).sum()),
                len(valid))
    if unparsed > 0:
        logger.info("  Unparsed:  %d/%d", unparsed, len(results_df))
    logger.info("=" * 64)

    out = output or f"step4_{name}_{task}_{explainer}_{shots_per_class}spc_results.csv"
    results_df.to_csv(out, index=False)
    logger.info("  Saved to %s", out)

    return results_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 4: ICL inference with pre-computed rationales")
    p.add_argument("--train", required=True,
                    help="Train CSV (ambiguous/difficult samples, used as shots)")
    p.add_argument("--test", required=True, help="Test CSV")
    p.add_argument("--rationales", required=True,
                    help="Rationales JSON from step 3")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--explainer", default="deeplift",
                choices=["deeplift", "phcot", "both"])
    p.add_argument("--task", default="offensive_en",
                    choices=["offensive_en", "irony_it"])
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--shots_per_class", type=int, default=10,
                    help="Number of ICL shots per class (total shots = shots_per_class × num_classes)")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    run_step4(
        args.train, args.test, args.rationales,
        args.model, args.explainer, args.task,
        args.top_k, args.n_steps,
        args.max_new_tokens, args.shots_per_class,
        args.device, args.output,
    )
