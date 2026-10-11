from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / "outputs" / "multimodal_affective_papers").resolve()
PAPERS = (OUT / "papers").resolve()
MANIFEST = OUT / "download_manifest.json"
ARCHIVE = ROOT / "outputs" / "multimodal_affective_papers.zip"

CATEGORIES = {
    "01_综述": [1],
    "02_表示学习与融合": [3, 7, 9],
    "03_缺失模态与补全": [2, 4, 5, 10, 11, 12, 13, 14, 15, 21],
    "04_模态可靠性与冲突": [6, 16, 17, 18],
    "05_评测与基准": [19, 20],
    "06_情绪原因与解释": [8],
}
BY_NUMBER = {number: category for category, numbers in CATEGORIES.items() for number in numbers}

records = json.loads(MANIFEST.read_text(encoding="utf-8"))
assert {record["number"] for record in records} == set(BY_NUMBER)

for category in CATEGORIES:
    folder = (PAPERS / category).resolve()
    assert folder.is_relative_to(PAPERS)
    folder.mkdir(parents=True, exist_ok=True)

for record in records:
    number = record["number"]
    category = BY_NUMBER[number]
    record["category"] = category
    if record["status"] != "downloaded":
        continue

    filename = Path(record["file"]).name
    old_path = (PAPERS / filename).resolve()
    new_path = (PAPERS / category / filename).resolve()
    assert old_path.is_relative_to(PAPERS) and new_path.is_relative_to(PAPERS)
    if old_path.exists():
        if new_path.exists():
            raise RuntimeError(f"目标文件已存在，未移动：{new_path}")
        old_path.rename(new_path)
    if not new_path.exists():
        raise RuntimeError(f"缺少下载文件：{new_path}")
    digest = hashlib.sha256(new_path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise RuntimeError(f"SHA-256 不匹配：{new_path}")
    record["relative_path"] = new_path.relative_to(OUT).as_posix()

(PAPERS / "01_综述" / "README.md").write_text(
    "# 综述\n\n01 Zhang et al. (2024) 的原文需要机构访问。详情及 DOI 见总目录。\n",
    encoding="utf-8",
)
(PAPERS / "02_表示学习与融合" / "03_获取说明.md").write_text(
    "# 03 Yang et al. (2026)\n\nCVPR 2026 会议日程已核对题名与作者，但 CVF 论文 PDF 链接目前返回 404。详情见总目录。\n",
    encoding="utf-8",
)

lines = [
    "# 多模态情感计算文献目录",
    "",
    "整理日期：2026-10-01。编号 01–09 对应最初提供的文献，10–21 对应后续调研补充的文献。",
    "已下载的 PDF 来自 CVF Open Access、ACL Anthology、NeurIPS、AAAI 或 arXiv，并已检查文件头、页数、首页文字和 SHA-256。",
    "",
    "## 文件夹分类",
    "",
    "| 文件夹 | 内容 |",
    "|---|---|",
    "| `01_综述` | 综述文献的获取说明；01 的 PDF 需要机构权限 |",
    "| `02_表示学习与融合` | 共享/特有表示、专家融合、模态分布对齐；含 03 的获取说明 |",
    "| `03_缺失模态与补全` | 蒸馏、表示补偿、生成补全、选择性修复 |",
    "| `04_模态可靠性与冲突` | 模态权重、可信度、冲突、因果不变表示 |",
    "| `05_评测与基准` | 噪声和不均衡缺失评测 |",
    "| `06_情绪原因与解释` | 对话情绪原因定位及摘要 |",
    "",
    "## 文献清单",
    "",
    "| 编号 | 论文 | 出处 | 分类 | 本地 PDF / 获取情况 |",
    "|---:|---|---|---|---|",
]
for record in sorted(records, key=lambda x: x["number"]):
    number = record["number"]
    title = record["title"]
    link = record["page_url"]
    venue = record["venue"]
    category = record["category"]
    if record["status"] == "downloaded":
        status = f"[PDF]({record['relative_path']})，{record['pages']} 页"
    elif number == 1:
        status = "期刊全文需机构访问；见本分类中的获取说明"
    elif number == 3:
        status = "CVF PDF 当前返回 404；见本分类中的获取说明"
    else:
        status = record.get("error", record["status"])
    lines.append(f"| {number:02d} | [{title}]({link}) | {venue} | {category} | {status} |")
lines.extend([
    "",
    "## 核对说明",
    "",
    "- 02 的 CVF 网页题名使用 *Hypergraph Conditioned Diffusion*，CVF PDF 首页使用 *Hypergraph Guided Diffusion*；最终参考文献请以正式引文页核对。",
    "- 03 的题名与作者已由 [CVPR 2026 会议日程](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf)核对，PDF 下载地址目前失效。",
    "- 20、21 是 arXiv 预印本，不在本目录中标记为正式发表。",
    "- 下载地址、页数、大小及 SHA-256 记录在 `download_manifest.json`。",
    "",
])
(OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(ROOT / "outputs"))

with zipfile.ZipFile(ARCHIVE) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f"ZIP 校验失败：{bad}")

print(json.dumps({
    "categories": {category: sum(r["status"] == "downloaded" and r["category"] == category for r in records) for category in CATEGORIES},
    "downloaded": sum(r["status"] == "downloaded" for r in records),
    "zip_files": len(zipfile.ZipFile(ARCHIVE).namelist()),
    "zip_bytes": ARCHIVE.stat().st_size,
}, ensure_ascii=False, indent=2))
