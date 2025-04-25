import os
import io
import requests
from typing import List, Tuple, Dict, Optional

import fitz  # PyMuPDF – 精准提取文本
from openai import OpenAI
from mcp.memory import MemoryStore

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------- 全局配置 ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, str] = {
    "llama2_agent":   "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}

# =============================================================
class LLMScheduler:
    """PDF 分发调度器（按页）。"""

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict] = []

    # ---------- PDF → 逐页文本 ----------
    @staticmethod
    def _pdf_pages(pdf_bytes: bytes) -> List[Tuple[str, str]]:
        pages: List[Tuple[str, str]] = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for idx, page in enumerate(doc, 1):
            txt = page.get_text("text") or ""
            if not txt.strip() and OCR_AVAILABLE:
                img = convert_from_bytes(pdf_bytes, dpi=300, first_page=idx, last_page=idx)[0]
                txt = pytesseract.image_to_string(img, lang="eng+chi_sim")
            pages.append((f"page_{idx}", txt.strip()))
        return pages

    # ---------- GPT‑4o 规划 ----------
    def _plan(self, task: str, page_ids: List[str]) -> Dict[str, List[str]]:
        ids_str = ", ".join(page_ids) if page_ids else "(空)"
        sys_msg = (
            "你是任务调度专家，请根据【任务】分配页面给以下 agent："
            + ", ".join(REGISTRY.keys()) + "。只返回分配结果，每行格式：agent: id1,id2 或 agent: m-n。"
        )
        user_msg = f"【任务】{task}\n【页面列表】{ids_str}"
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        )
        return self._parse_plan(rsp.choices[0].message.content)

    @staticmethod
    def _parse_plan(text: str) -> Dict[str, List[str]]:
        plan = {k: [] for k in REGISTRY}
        for line in text.splitlines():
            if ':' not in line:
                continue
            ag, ids = line.split(':', 1)
            ag = ag.strip()
            if ag not in REGISTRY:
                continue
            ids = ids.replace(' ', '')
            for part in ids.split(','):
                if not part:
                    continue
                if '-' in part:
                    start, end = part.split('-')
                    s_idx = int(''.join(filter(str.isdigit, start)))
                    e_idx = int(''.join(filter(str.isdigit, end)))
                    prefix = ''.join(filter(str.isalpha, start)) or 'page_'
                    plan[ag] += [f"{prefix}{i}" for i in range(s_idx, e_idx + 1)]
                else:
                    plan[ag].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- 主入口 ----------
    def dispatch(
        self,
        context_id: str,
        task: str,
        pdf_bytes: Optional[bytes] = None,
        file_bytes: Optional[bytes] = None,
    ):
        """兼容 server.py 传参：既支持 pdf_bytes，也支持 file_bytes。"""
        pdf_data = pdf_bytes or file_bytes
        if not pdf_data:
            raise ValueError("dispatch 需要提供 pdf_bytes 或 file_bytes")

        pages = self._pdf_pages(pdf_data)
        page_map = dict(pages)
        plan = self._plan(task, [pid for pid, _ in pages])

        # —— 分发到本地 Agent ——
        for ag, pid_list in plan.items():
            snapshot = self.memory.get(context_id)
            for pid in pid_list:
                context_txt = page_map.get(pid, '')
                prompt_text = f"【任务指令】{task}\n\n【{pid} 原文】\n{context_txt}"
                payload = {
                    "prompt": prompt_text,
                    "context_id": context_id,
                    "page_id": pid,
                    "agent_name": ag,
                    "shared_memory": snapshot
                }
                try:
                    resp = requests.post(REGISTRY[ag], json=payload, timeout=120)
                    data = resp.json()
                    mem_update = data.get('memory_update') or {ag: {pid: data.get('result', '')}}
                except Exception as e:
                    self.trace.append({"agent": ag, "subtask": pid, "output": f"[❌ 调用失败] {e}"})
                    continue

                # 更新 memory
                current = self.memory.get(context_id).get(ag, {})
                if not isinstance(current, dict):
                    current = {}
                current.update(mem_update.get(ag, {}))
                self.memory.update(context_id, {ag: current})
                self.trace.append({"agent": ag, "subtask": pid, "output": current.get(pid, '')})

        # —— GPT‑4o 每页摘要 ——
        page_summaries: Dict[str, str] = {}
        for pid, _ in pages:
            page_analysis = ''
            for ag in plan:
                ag_mem = self.memory.get(context_id).get(ag, {})
                if pid in ag_mem:
                    page_analysis = ag_mem[pid]
                    break
            if not page_analysis:
                continue
            rsp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": f"请总结：\n{page_analysis}"}])
            page_summaries[pid] = rsp.choices[0].message.content.strip()
        self.memory.update(context_id, {"page_summaries": page_summaries})

        # —— GPT‑4o 全文总结 ——
        overall_prompt = "下面是 PDF 每页摘要，请生成整体总结（共识/差异/建议）：\n\n"
        for pid in sorted(page_summaries, key=lambda x: int(x.split('_')[1])):
            overall_prompt += f"【{pid} 摘要】{page_summaries[pid]}\n"
        rsp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": overall_prompt}])
        final_summary = rsp.choices[0].message.content.strip()

        self.memory.update(context_id, {"summary": final_summary})
        self.trace.append({"agent": "scheduler", "subtask": "自动生成全文总结", "output": final_summary})
        return self.memory.get(context_id), self.trace

