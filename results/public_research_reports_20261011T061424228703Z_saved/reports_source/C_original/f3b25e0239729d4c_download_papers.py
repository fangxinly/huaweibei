from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "multimodal_affective_papers"
PAPERS_DIR = OUT / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = [
    (1, "Zhang_2024_Deep_Learning_MER_Review", "Deep learning-based multimodal emotion recognition from audio, visual, and text modalities: A systematic review of recent advancements and future prospects", "Expert Systems with Applications 2024", "综述", None, "https://doi.org/10.1016/j.eswa.2023.121692"),
    (2, "Qiu_2026_HyperEF", "Beyond Missing Modalities: Hypergraph Conditioned Diffusion for Uncertainty-Aware Multimodal Emotion Recognition", "CVPR 2026", "缺失模态与不确定性", "https://openaccess.thecvf.com/content/CVPR2026/papers/Qiu_Beyond_Missing_Modalities_Hypergraph_Conditioned_Diffusion_for_Uncertainty-Aware_Multimodal_Emotion_CVPR_2026_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_Beyond_Missing_Modalities_Hypergraph_Conditioned_Diffusion_for_Uncertainty-Aware_Multimodal_Emotion_CVPR_2026_paper.html"),
    (3, "Yang_2026_FUSE_Net", "Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis", "CVPR 2026", "融合与表示", "https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf"),
    (4, "Zhuang_2025_CMAD", "CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities", "ICCV 2025", "蒸馏与缺失模态", "https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf", "https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html"),
    (5, "Zhu_2025_P_RMF", "Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data", "ACL 2025", "代理表示与鲁棒性", "https://aclanthology.org/2025.acl-long.1075.pdf", "https://aclanthology.org/2025.acl-long.1075/"),
    (6, "Mai_2026_CmIR", "Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective", "ACL 2026", "因果不变表示", "https://aclanthology.org/2026.acl-long.2119.pdf", "https://aclanthology.org/2026.acl-long.2119/"),
    (7, "Fang_2025_EMOE", "EMOE: Modality-Specific Enhanced Dynamic Emotion Experts", "CVPR 2025", "融合与表示", "https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html"),
    (8, "Wan_2026_MECES", "Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation", "ACL 2026", "情绪原因与解释", "https://aclanthology.org/2026.acl-long.2012.pdf", "https://aclanthology.org/2026.acl-long.2012/"),
    (9, "Mai_2026_CaReFlow", "CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion", "CVPR 2026", "融合与表示", "https://openaccess.thecvf.com/content/CVPR2026/papers/Mai_CaReFlow_Cyclic_Adaptive_Rectified_Flow_for_Multimodal_Fusion_CVPR_2026_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2026/html/Mai_CaReFlow_Cyclic_Adaptive_Rectified_Flow_for_Multimodal_Fusion_CVPR_2026_paper.html"),
    (10, "Zhao_2021_MMIN", "Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities", "ACL 2021", "缺失模态基础方法", "https://aclanthology.org/2021.acl-long.203.pdf", "https://aclanthology.org/2021.acl-long.203/"),
    (11, "Lin_2023_MissModal", "MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis", "TACL 2023", "缺失模态基础方法", "https://aclanthology.org/2023.tacl-1.94.pdf", "https://aclanthology.org/2023.tacl-1.94/"),
    (12, "Li_2024_CorrKD", "Correlation-Decoupled Knowledge Distillation for Multimodal Sentiment Analysis with Incomplete Modalities", "CVPR 2024", "蒸馏与缺失模态", "https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Correlation-Decoupled_Knowledge_Distillation_for_Multimodal_Sentiment_Analysis_with_Incomplete_Modalities_CVPR_2024_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Correlation-Decoupled_Knowledge_Distillation_for_Multimodal_Sentiment_Analysis_with_Incomplete_Modalities_CVPR_2024_paper.pdf"),
    (13, "Li_2024_HRLF", "Toward Robust Incomplete Multimodal Sentiment Analysis via Hierarchical Representation Learning", "NeurIPS 2024", "表示学习与缺失模态", "https://proceedings.neurips.cc/paper_files/paper/2024/file/3209cf3312b2cbb68e33644362ddc2bd-Paper-Conference.pdf", "https://proceedings.neurips.cc/paper_files/paper/2024/file/3209cf3312b2cbb68e33644362ddc2bd-Paper-Conference.pdf"),
    (14, "Dai_2025_MD2N", "Unbiased Missing-modality Multimodal Learning", "ICCV 2025", "生成补全与偏差", "https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_Unbiased_Missing-modality_Multimodal_Learning_ICCV_2025_paper.pdf", "https://www.openaccess.thecvf.com/content/ICCV2025/html/Dai_Unbiased_Missing-modality_Multimodal_Learning_ICCV_2025_paper.html"),
    (15, "Zhuang_2025_HME", "Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities", "NeurIPS 2025", "跨样本增强与缺失模态", "https://proceedings.neurips.cc/paper_files/paper/2025/file/d079de28c5e3f22e3507db24e870126b-Paper-Conference.pdf", "https://proceedings.neurips.cc/paper_files/paper/2025/hash/d079de28c5e3f22e3507db24e870126b-Abstract-Conference.html"),
    (16, "He_2026_EBMC", "Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis", "CVPR 2026", "模态平衡与可信度", "https://openaccess.thecvf.com/content/CVPR2026/papers/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html"),
    (17, "Wang_2026_CACR", "Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment Analysis", "CVPR 2026", "模态冲突", "https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Conflict-Aware_Adaptive_Cross-Reconstruction_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Conflict-Aware_Adaptive_Cross-Reconstruction_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html"),
    (18, "Jiang_2026_CICA", "CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis", "CVPR 2026", "可信度与鲁棒融合", "https://openaccess.thecvf.com/content/CVPR2026/papers/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.pdf", "https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html"),
    (19, "Mao_2023_Robust_MSA", "Robust-MSA: Understanding the Impact of Modality Noise on Multimodal Sentiment Analysis", "AAAI 2023 Demo", "噪声评测", "https://ojs.aaai.org/index.php/AAAI/article/download/27078/26850", "https://ojs.aaai.org/index.php/AAAI/article/view/27078"),
    (20, "Pham_2026_MissBench_PREPRINT", "MissBench: Benchmarking Multimodal Affective Analysis under Imbalanced Missing Modalities", "arXiv 2026（预印本）", "缺失机制评测", "https://arxiv.org/pdf/2603.09874", "https://arxiv.org/abs/2603.09874"),
    (21, "Gao_2026_SIEVE_PREPRINT", "Should Missing Modalities Always Be Necessary to Repair for Multi-modal Sentiment Analysis?", "arXiv 2026（预印本）", "选择性补全", "https://arxiv.org/pdf/2607.17262", "https://arxiv.org/abs/2607.17262"),
]


def fetch(paper):
    number, stem, title, venue, topic, pdf_url, page_url = paper
    if pdf_url is None:
        return {"number": number, "status": "no_open_pdf", "error": "期刊页面显示需机构访问或购买；未找到可信的作者公开版本。"}
    path = PAPERS_DIR / f"{number:02d}_{stem}.pdf"
    if path.exists():
        data = path.read_bytes()
    else:
        last_error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (research literature collection)", "Accept": "application/pdf,*/*"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    data = response.read()
                if not data.startswith(b"%PDF-"):
                    raise ValueError(f"不是 PDF：{data[:80]!r}")
                path.write_bytes(data)
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(attempt + 1)
        else:
            return {"number": number, "status": "failed", "error": last_error}
    try:
        reader = PdfReader(str(path), strict=False)
        pages = len(reader.pages)
        first_page = reader.pages[0].extract_text() or ""
        if pages < 2 or len(first_page.strip()) < 50:
            raise ValueError("PDF 页数或首页可提取文字异常")
        size = path.stat().st_size
        sha = hashlib.sha256(data).hexdigest()
        return {"number": number, "status": "downloaded", "file": path.name, "pages": pages, "bytes": size, "sha256": sha, "first_page_excerpt": re.sub(r"\s+", " ", first_page)[:120]}
    except Exception as exc:
        path.unlink(missing_ok=True)
        return {"number": number, "status": "invalid", "error": str(exc)}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch, PAPERS))

by_number = {result["number"]: result for result in results}
lines = [
    "# 多模态情感计算文献目录",
    "",
    "整理日期：2026-10-01。编号 01–09 对应最初提供的文献，10–21 对应后续调研补充的文献。",
    "PDF 优先取自 CVF Open Access、ACL Anthology、NeurIPS、AAAI；预印本取自 arXiv。文件通过 PDF 文件头、页数和首页文字检查。",
    "",
    "| 编号 | 论文 | 出处 | 主题 | 本地 PDF / 获取情况 |",
    "|---:|---|---|---|---|",
]
for paper in PAPERS:
    number, stem, title, venue, topic, pdf_url, page_url = paper
    result = by_number[number]
    local = f"[下载文件](papers/{result['file']})，{result['pages']} 页" if result["status"] == "downloaded" else result.get("error", result["status"])
    lines.append(f"| {number:02d} | [{title}]({page_url}) | {venue} | {topic} | {local} |")
lines += [
    "",
    "## 获取说明",
    "",
    "- 01 为期刊付费访问文献。本包仅提供 [期刊 DOI 页面](https://doi.org/10.1016/j.eswa.2023.121692)，没有用其他论文替代，也没有绕过访问限制。",
    "- 02 的 CVF 网页题名使用 *Hypergraph Conditioned Diffusion*，下载的 CVF PDF 首页题名使用 *Hypergraph Guided Diffusion*；请以正式引文页核对最终参考文献。",
    "- 03 已由 [CVPR 2026 官方会议日程](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf)核对题名和作者，但检索到的 CVF PDF 地址目前返回 404；目录保留该地址以便之后重试。",
    "- 20、21 为预印本，尚未在本目录中标记为正式会议或期刊论文。",
    "- SHA-256、大小和页数记录在 `download_manifest.json`，用于后续核对文件完整性。",
    "",
]
(OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
manifest = []
for paper in PAPERS:
    number, stem, title, venue, topic, pdf_url, page_url = paper
    manifest.append({"number": number, "title": title, "venue": venue, "topic": topic, "pdf_url": pdf_url, "page_url": page_url, **by_number[number]})
(OUT / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

archive = ROOT / "outputs" / "multimodal_affective_papers.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(ROOT / "outputs"))

print(json.dumps({"downloaded": sum(r["status"] == "downloaded" for r in results), "not_downloaded": [r for r in results if r["status"] != "downloaded"], "archive": str(archive), "zip_bytes": archive.stat().st_size}, ensure_ascii=False, indent=2))
