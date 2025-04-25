import os
import io
import requests
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
from openai import OpenAI
from mcp.memory import MemoryStore

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, str] = {
    "llama2_agent": "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}

class LLMScheduler:
    """按 PDF 页面拆分；多页结果累积到 memory 列表，汇总时一次性交给 GPT-4o。"""

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict] = []

    # ---------- PDF 解析 ----------
    def _pdf_pages(self, pdf_bytes: bytes):
        pages = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for idx, page in enumerate(doc, 1):
            txt = page.get_text("text") or ""
            if not txt.strip() and OCR_AVAILABLE:
                img = convert_from_bytes(pdf_bytes, dpi=300, first_page=idx, last_page=idx)[0]
                txt = pytesseract.image_to_string(img, lang="eng+chi_sim")
            pages.append((f"page_{idx}", txt.strip()))
        return pages

    # ---------- GPT 分配 ----------
    def _plan_with_gpt(self, task: str, page_ids: List[str]):
        ids = ", ".join(page_ids) if page_ids else "(无)"
        sys = (
            "你是任务调度专家，根据【任务】和【页面列表】挑选相关页面并分配给下列 agent："
            + ", ".join(REGISTRY.keys()) + "。仅输出分配结果，格式 agent: id1,id2 或 agent: m-n。"
        )
        user = f"【任务】\n{task}\n\n【页面列表】{ids}"
        rsp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":sys},{"role":"user","content":user}])
        return self._parse_plan(rsp.choices[0].message.content)

    @staticmethod
    def _parse_plan(txt: str):
        plan = {k: [] for k in REGISTRY}
        for ln in txt.splitlines():
            if ':' not in ln: continue
            ag, ids = ln.split(':',1)
            ag = ag.strip()
            if ag not in REGISTRY: continue
            ids = ids.replace(' ','')
            for part in ids.split(','):
                if not part: continue
                if '-' in part:
                    s,e = part.split('-')
                    s_num = int(''.join(filter(str.isdigit,s)))
                    e_num = int(''.join(filter(str.isdigit,e)))
                    prefix = ''.join(filter(str.isalpha,s)) or 'page_'
                    plan[ag]+= [f"{prefix}{i}" for i in range(s_num,e_num+1)]
                else:
                    plan[ag].append(part)
        return {k:v for k,v in plan.items() if v}

    # ---------- 主调度 ----------
    def dispatch(self, context_id: str, task: str, file_bytes: bytes|None=None):
        pages = self._pdf_pages(file_bytes) if file_bytes else []
        page_map = dict(pages)
        plan = self._plan_with_gpt(task,[pid for pid,_ in pages])

        for ag, pid_list in plan.items():
            snapshot = self.memory.get(context_id)
            for pid in pid_list:
                prompt = (
                    f"【任务指令】{task}\n\n【{pid} 原文】\n{page_map.get(pid,'')}"
                )
                payload = {
                    "prompt": prompt,
                    "context_id": context_id,
                    "page_id": pid,
                    "agent_name": ag,
                    "shared_memory": snapshot
                }
                try:
                    r = requests.post(REGISTRY[ag], json=payload, timeout=120)
                    jr = r.json()
                    clean = jr.get('result','').replace('[INST]','').replace('[/INST]','').strip()
                    mem_list = self.memory.get(context_id).get(ag, [])
                    if not isinstance(mem_list, list):
                        mem_list = [mem_list] if mem_list else []
                    mem_list.append(f"## {pid}\n{clean}")
                    self.memory.update(context_id,{ag: mem_list})
                    self.trace.append({"agent":ag,"subtask":pid,"output":clean})
                except Exception as e:
                    self.trace.append({"agent":ag,"subtask":pid,"output":f"[❌ 调用失败] {e}"})

        # ---------- 汇总 ----------
        summary_prompt = "请根据以下各 agent 的全部页面分析，撰写总结：\n\n"
        for ag in plan:
            segs = self.memory.get(context_id).get(ag, [])
            content = "\n\n".join(segs) if isinstance(segs, list) else segs
            summary_prompt += f"【{ag}】\n{content}\n\n"
        rsp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":summary_prompt}])
        summary = rsp.choices[0].message.content.strip()
        self.memory.update(context_id,{"summary":summary})
        self.trace.append({"agent":"scheduler","subtask":"自动生成总结","output":summary})
        return self.memory.get(context_id), self.trace

