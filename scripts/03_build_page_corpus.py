"""按页提取 T8 说明书，并将常见问题表格按行保存。"""
import json
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data/raw/roborock/Roborock_T8_User_Manual_ZH.pdf"
OUTPUT = ROOT / "data/processed/t8_pages.jsonl"

def clean_cell(text):
    return "".join((text or "").split())

def records_for_page(page, page_number):
    if page_number == 23:
        records = []
        for table in page.extract_tables():
            for problem, solution in table:
                if problem == "问题类型" or not problem or not solution:
                    continue
                problem, solution = clean_cell(problem), clean_cell(solution)
                records.append({"model": "石头 T8", "pdf_page": 23,
                                "section": "常见问题", "title": problem,
                                "text": f"问题：{problem}。解决方法：{solution}"})
        return records
    if page_number in (3, 12, 15):
        middle = page.width / 2
        top, bottom = 20, page.height - 20

        left = page.crop(
            (0, top, middle, bottom)
        ).extract_text() or ""

        right = page.crop(
            (middle, top, page.width, bottom)
        ).extract_text() or ""

        text = left + "\n" + right
    else:
        text = page.extract_text() or ""
    text = text.strip()
    section = text.splitlines()[0] if text else ""
    return [{"model": "石头 T8", "pdf_page": page_number,
             "section": section, "title": "", "text": text}]

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with pdfplumber.open(PDF) as pdf, OUTPUT.open("w", encoding="utf-8") as out:
        for page_number, page in enumerate(pdf.pages, start=1):
            for record in records_for_page(page, page_number):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"已生成 {count} 条知识记录，来源 PDF 共 {len(pdf.pages)} 页。")

if __name__ == "__main__":
    main()
