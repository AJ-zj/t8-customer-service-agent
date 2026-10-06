"""基础关键词检索：数资料里命中了几个关键词。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/t8_pages.jsonl"


def main():
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    documents = [json.loads(line) for line in lines]
    questions = [
        ("软胶主刷缠了头发怎么拆下来清理", ["软胶主刷", "缠绕"], 15),
        ("清扫时机器人有异响怎么办", ["清扫时有异响", "主刷", "缠绕"], 23),
    ]
    for question, keywords, expected_page in questions:
        results = []
        for document in documents:
            matched = [word for word in keywords if word in document["text"]]
            results.append((len(matched), document, matched))
        results.sort(key=lambda result: result[0], reverse=True)
        score, best, matched = results[0]
        print(f"问题：{question}")
        print(f"命中关键词：{matched}")
        if score == 0:
            print("没有找到相关知识")
            continue
        print(f"找到第 {best['pdf_page']} 页「{best['section']}」，共命中 {score} 个词")
        print(f"核对预期第 {expected_page} 页：{best['pdf_page'] == expected_page}")
        print(best["text"][:120], "\n")


if __name__ == "__main__":
    main()
