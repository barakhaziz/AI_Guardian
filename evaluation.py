# evaluate_beavertails.py
"""A small utility script to benchmark a safety‑judging system against the
BeaverTails dataset and display a confusion matrix.

Usage (from the command line or a notebook):

    python evaluate_beavertails.py                   # quick sanity‑check on 1 000 samples
    python evaluate_beavertails.py --split 330k_test # full evaluation on the 33 k test set

Inside another script / notebook you can also do

    from evaluate_beavertails import evaluate
    metrics = evaluate(judge, split="330k_test", limit=None)

The script assumes you already have a class `SafetyJudge` (or equivalent)
that implements a method

    judge(user_input: str, assistant_output: str | None = None) -> str

returning the string "safe" or "unsafe".
Replace the stub `DummyJudge` below with your own implementation.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter

from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

###############################################################################
# 1.  Your safety‑judging class
###############################################################################

class DummyJudge:
    """A placeholder judge that always says *unsafe* for demonstration only.
    Replace this with your own class or wrap calls to your remote model.
    """

    def judge(self, user_input: str, assistant_output: str | None = None) -> str:
        _ = (user_input, assistant_output)  # ignore inputs
        return "unsafe"

###############################################################################
# 2.  Evaluation logic
###############################################################################

LABELS = ["safe", "unsafe"]  # fixed order for confusion‑matrix axes


def evaluate(
    judge: "DummyJudge | SafetyJudge",
    split: str = "330k_train",
    limit: int | None = 1000,  # default to 1k for a quick run – set to None for full split
    conversation: bool = True,  # if True, pass both prompt & response to judge
):
    """Run *judge* over the selected BeaverTails split and return summary dict."""

    print(f"Loading BeaverTails split '{split}' …", file=sys.stderr)
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)

    if limit:
        ds = ds.select(range(limit))
        print(f"Evaluating on a subset of {limit} samples for speed.", file=sys.stderr)

    y_true: list[str] = []
    y_pred: list[str] = []
    y_pred_single: list[str] = []  # for single-prompt evaluation

    for row in ds:
        prompt: str = row["prompt"]
        response: str | None = row.get("response")  # some splits omit this

        gold = "safe" if bool(row["is_safe"]) else "unsafe"
        pred = judge.judge(prompt, response if conversation else None, single_prompt=False)  # for conversation evaluation
        pred_single = judge.judge(prompt, response if conversation else None, single_prompt=True)  # for single-prompt evaluation

        # Normalise just in case
        pred = pred.strip().lower()
        if pred not in LABELS:
            raise ValueError(
                f"Judge returned '{pred}', expected one of {LABELS}.")

        y_true.append(gold)
        y_pred.append(pred)
        y_pred_single.append(pred_single)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_single = confusion_matrix(y_true, y_pred_single, labels=LABELS)

    print("\n=== Classification report of Messages ===")
    print(classification_report(y_true, y_pred, target_names=LABELS))
    print("Overall accuracy:", f"{acc*100:.2f}%")

    print("\n=== Classification report of Single Prompt ===")
    print(classification_report(y_true, y_pred_single, target_names=LABELS))
    acc_single = accuracy_score(y_true, y_pred_single)
    print("Overall accuracy (single prompt):", f"{acc_single*100:.2f}%")

    # Plot confusion matrix
    print("\n=== Confusion matrix ===")
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)),
           xticklabels=LABELS, yticklabels=LABELS,
           ylabel="True label", xlabel="Predicted label",
           title="BeaverTails confusion matrix")

    # Annotate cells
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    fig_single, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_single, interpolation="nearest")  # <-- use cm_single
    fig_single.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(LABELS)), yticks=range(len(LABELS)),
        xticklabels=LABELS, yticklabels=LABELS,
        ylabel="True label", xlabel="Predicted label",
        title="BeaverTails confusion – Single Prompt",
    )
    thresh = cm_single.max() / 2                         # <-- use cm_single
    for i, j in itertools.product(range(cm_single.shape[0]), range(cm_single.shape[1])):
        ax.text(j, i, format(cm_single[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_single[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

###############################################################################
# 3.  CLI entry‑point
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a safety judge on BeaverTails.")
    parser.add_argument("--split", default="30k_test",
                        help="Dataset split to use (default: 30k_test)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Sample limit for a quick run (default: 1000; use 0 for full)")
    parser.add_argument(
        "--no-conversation",
        action="store_false",
        dest="conversation",
        help="Judge both user prompt and assistant response (default: only prompt)",
    )
    args = parser.parse_args()

    # Instantiate *your* judge here
    from src.detecting_attacks.llama_guard import JudgeLLM  # or your own judge class
    from dotenv import load_dotenv
    load_dotenv(override=True)
    import os
    guard_model = os.getenv("GUARD_MODEL", "llama-guard3")  # ensure the model is set
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")  # ensure the Ollama URL is set
    judge = JudgeLLM(model_name=guard_model, ollama_url=ollama_url)
    import time
    start_time = time.time()
    print(args.conversation)
    evaluate(judge,
             split=args.split,
             limit=args.limit,
             conversation=args.conversation)
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")