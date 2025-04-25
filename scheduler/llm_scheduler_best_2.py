import os
import io
import requests
from typing import List, Tuple, Dict

import fitz  # PyMuPDF，用于精准解析数字 PDF
from openai import OpenAI
from mcp.memory import MemoryStore

# ---------- OCR 备份方案（可选） ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------- OpenAI 客户端 ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- 本地 Agent 注册表 ----------
REGISTRY: Dict[str, str] = {
    "llama2_agent": "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}

# =========================================================
#                       Scheduler
# =========================================================
class LLMScheduler:
    """按 PDF **页面** 拆分；优先用 PyMuPDF 提取文字，提取失败时自动 OCR。"""

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict] = []

    # ---------- 1. 解析 PDF 每页文本 ----------
    def _pdf_pages(self, pdf_bytes: bytes) -> List[Tuple[str, str]]:
        pages: List[Tuple[str, str]] = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for idx, page in enumerate(doc, 1):
            text = page.get_text("text") or ""
            # 若纯空白且允许 OCR，尝试识别
            if not text.strip() and OCR_AVAILABLE:
                img = convert_from_bytes(pdf_bytes, dpi=300, first_page=idx, last_page=idx)[0]
                text = pytesseract.image_to_string(img, lang="eng+chi_sim")
            pages.append((f"page_{idx}", text.strip()))
        return pages

    # ---------- 2. 让 GPT‑4o 根据页面 ID 分配 ----------
    def _plan_with_gpt(self, user_text: str, page_ids: List[str]) -> Dict[str, List[str]]:
        id_list = ", ".join(page_ids) if page_ids else "(无页面)"
        sys_prompt = (
            "你是任务调度专家。请根据【任务全文】和【PDF 页面列表】判断哪些页面与任务相关，"
            "并将相关页面分配给以下 agent：" + ", ".join(REGISTRY.keys()) + "。\n\n"
            "仅输出分配结果，格式：agent: id1,id2 或 agent: start-end。"
        )
        user_prompt = f"【任务全文】\n{user_text}\n\n【页面列表】{id_list}"
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        )
        return self._parse_plan(rsp.choices[0].message.content)

    @staticmethod
    def _parse_plan(text: str) -> Dict[str, List[str]]:
        plan = {k: [] for k in REGISTRY}
        for line in text.splitlines():
            if ":" not in line:
                continue
            agent, ids = line.split(":", 1)
            agent = agent.strip()
            if agent not in REGISTRY:
                continue
            ids = ids.replace(" ", "")
            for part in ids.split(','):
                if not part:
                    continue
                if '-' in part:
                    start, end = part.split('-')
                    prefix = ''.join(filter(str.isalpha, start)) or 'page_'
                    s, e = int(''.join(filter(str.isdigit, start))), int(''.join(filter(str.isdigit, end)))
                    plan[agent] += [f"{prefix}{i}" for i in range(s, e + 1)]
                else:
                    plan[agent].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- 3. 主调度 ----------
    def dispatch(self, context_id: str, text: str, file_bytes: bytes | None = None):
        pages = self._pdf_pages(file_bytes) if file_bytes else []
        page_map = dict(pages)
        plan = self._plan_with_gpt(text, [pid for pid, _ in pages])

        for agent, pid_list in plan.items():
            snapshot = self.memory.get(context_id)
            for pid in pid_list:
                page_text = page_map.get(pid, "")
                prompt = (
                    "你是一名 GAIA benchmark 文档分析专家，请基于任务指令和对应页面内容回答：\n"
                    f"【任务指令】{text}\n\n【{pid} 原文】\n{page_text}"
                )
                payload = {
                    "prompt": prompt,
                    "context_id": context_id,
                    "page_id": pid,
                    "agent_name": agent,
                    "shared_memory": snapshot
                }
                try:
                    resp = requests.post(REGISTRY[agent], json=payload, timeout=120)
                    jr = resp.json()
                    clean = jr.get("result", "").replace("[INST]", "").replace("[/INST]", "").strip()
                    mem = jr.get("memory_update") or {agent: clean}
                except Exception as e:
                    self.trace.append({"agent": agent, "subtask": pid, "output": f"[❌ 调用失败] {e}"})
                    continue

                self.memory.update(context_id, mem)
                self.trace.append({"agent": agent, "subtask": pid, "output": mem.get(agent, "")})

        # ---------- 4. GPT‑4o 汇总 ----------
        summary_prompt = "请基于以下 agent 输出撰写总结（共识/差异/建议）：\n\n"
        for ag in plan:
            summary_prompt += f"【{ag}】{self.memory.get(context_id).get(ag, '')}\n\n"
        rsp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}])
        summary = rsp.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append({"agent": "scheduler", "subtask": "自动生成总结", "output": summary})

        return self.memory.get(context_id), self.trace

