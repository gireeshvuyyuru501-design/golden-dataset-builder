import json
import time
import random
from google.genai import errors
import os
import re
import string
import urllib.request
from collections import Counter
from pathlib import Path

from google import genai

DATA_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
DATA_PATH = Path("squad_validation.json")
REPORT_PATH = Path("evaluation_results.json")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def normalize(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return int(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference):
    predicted = normalize(prediction).split()
    expected = normalize(reference).split()

    if not predicted or not expected:
        return float(predicted == expected)

    overlap = sum((Counter(predicted) & Counter(expected)).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(expected)

    return 2 * precision * recall / (precision + recall)


def load_questions(limit=10):
    if not DATA_PATH.exists():
        print("Downloading the real SQuAD validation dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    questions = []

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                questions.append({
                    "id": qa["id"],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "references": [a["text"] for a in qa["answers"]],
                })

                if len(questions) >= limit:
                    return questions

    return questions


def evaluate():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client()
    dataset = load_questions(limit=10)
    results = []

    try:
        for index, item in enumerate(dataset, start=1):
            prompt = (
                "Answer the question using only the context. "
                "Return a short factual answer. "
                "If the context does not contain the answer, "
                "reply UNKNOWN.\n\n"
                f"Context:\n{item['context']}\n\n"
                f"Question: {item['question']}"
            )

            for attempt in range(5):
                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=prompt,
                    )
                    break
                except errors.ServerError as exc:
                    if exc.code not in (500, 502, 503, 504) or attempt == 4:
                        raise
                    delay = min(30, 2 ** attempt + random.random())
                    print(f"Temporary Gemini error {exc.code}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            prediction = (response.text or "").strip()

            em = max(
                exact_match(prediction, ref)
                for ref in item["references"]
            )

            f1 = max(
                token_f1(prediction, ref)
                for ref in item["references"]
            )

            results.append({
                **item,
                "prediction": prediction,
                "exact_match": em,
                "f1": round(f1, 4),
                "model": MODEL,
            })

            print(
                f"{index}/{len(dataset)} "
                f"EM={em} F1={f1:.3f}"
            )

    finally:
        client.close()

    report = {
        "dataset": "SQuAD v1.1 validation",
        "model": MODEL,
        "total_questions": len(results),
        "exact_match": round(
            sum(r["exact_match"] for r in results) / len(results), 4
        ) if results else None,
        "average_f1": round(
            sum(r["f1"] for r in results) / len(results), 4
        ) if results else None,
        "results": results,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEvaluation completed!")
    print("Exact Match:", report["exact_match"])
    print("Average F1:", report["average_f1"])
    print("Report:", REPORT_PATH.resolve())


if __name__ == "__main__":
    evaluate()
