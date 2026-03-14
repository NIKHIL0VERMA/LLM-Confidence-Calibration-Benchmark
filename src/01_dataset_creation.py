import pandas as pd
import re
from datasets import load_dataset

SAMPLE_SIZE = 6000

n_per_dataset = SAMPLE_SIZE // 3

# Textual dataset
truthfulqa = load_dataset("truthful_qa", "generation", split="validation").shuffle(seed=42)
truthfulqa = truthfulqa.select(range(min(len(truthfulqa), n_per_dataset)))

# common sense dataset
commonsenseqa = load_dataset("commonsense_qa", split='validation').shuffle(seed=42)
commonsenseqa = commonsenseqa.select(range(min(len(commonsenseqa), n_per_dataset)))

# binary dataset
boolq = load_dataset("boolq", split="validation").shuffle(seed=42)
boolq = boolq.select(range(min(len(boolq), n_per_dataset)))

# Numberical and mathematical dataset
gsm8k = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
gsm8k = gsm8k.select(range(min(len(gsm8k), n_per_dataset)))

questions = []

# TruthfulQA
for item in truthfulqa:
    questions.append({
        "dataset": "truthfulqa",
        "question": item["question"],
        "answer": item["best_answer"],
        "type": "text"
    })

# CommonSenseQA
for item in commonsenseqa:
    choices = item["choices"]
    labels = choices["label"]
    texts = choices["text"]

    options = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])

    answer_index = labels.index(item["answerKey"])
    answer_text = texts[answer_index]

    questions.append({
        "dataset": "commonsenseqa",
        "question": f"{item['question']}\n\n{options}",
        "answer": answer_text,
        "type": "text"
    })

# BoolQ
for item in boolq:
    questions.append({
        "dataset": "boolq",
        "question": f"{item['passage']}\n\nQ: {item['question']}",
        "answer": "true" if item["answer"] else "false",
        "type": "boolean"
    })

# GSM8k
for item in gsm8k:
    answer_match = re.search(r'####\s*(-?\d+\.?\d*)', item["answer"])
    answer = answer_match.group(1) if answer_match else item["answer"]
    questions.append({
        "dataset": "gsm8k",
        "question": item["question"],
        "answer": answer,
        "type": "numeric"
    })

df = pd.DataFrame(questions)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["id"] = df.index
df = df[["id", "dataset", "type", "question", "answer"]]

df.to_parquet("./data/Confidence_calibration_study_dataset.parquet", index=False)
df.to_csv("./data/Confidence_calibration_study_dataset.csv", index=False)