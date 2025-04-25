import os
import io
import requests
from typing import List, Tuple, Dict

import pdfplumber  # 更稳定的 PDF 文本抽取
from openai import OpenAI
from mcp.memory import MemoryStore

# 如需 OCR：确保系统已安装 tesseract，且 pip 安装 pdf2image、pytesseract
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
    """按 **PDF 页** 维度拆分，抽不到文本时自动 OCR。"""

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict] = []

    # ---------- PDF → [(page_id, text)] ----------
    @staticmethod
    def _pdf_pages(pdf_bytes: bytes) -> List[Tuple[str, str]]:
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # 若纯空白，尝试 OCR
                if not text.strip() and OCR_AVAILABLE:
                    img = convert_from_bytes(pdf_bytes, first_page=idx, last_page=idx)[0]
                    text = pytesseract.image_to_string(img, lang="eng+chi_sim")
                pages.append((f"page_{idx}", text.strip()))
        return pages

    # ---------- GPT‑4o 分配 ----------
    def _plan_with_gpt(self, user_text: str, page_ids: List[str]) -> Dict[str, List[str]]:
        id_list = ", ".join(page_ids) if page_ids else "(无页面)"
        sys_prompt = (
            "你是任务调度专家。请根据【用户任务全文】和【PDF 页面 ID 列表】判断哪些页面与任务相关，"
            "并将相关页面分配给以下 agent，格式：agent: id1,id2 或 agent: start-end。\n"
            + "可用 agent：" + ", ".join(REGISTRY.keys())
        )
        user_prompt = f"【用户任务全文】\n{user_text}\n\n【PDF 页面 ID 列表】{id_list}"
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        )
        return self._parse_plan(res.choices[0].message.content)

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
                    s = int(''.join(filter(str.isdigit, start)))
                    e = int(''.join(filter(str.isdigit, end)))
                    plan[agent] += [f"{prefix}{i}" for i in range(s, e + 1)]
                else:
                    plan[agent].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- 主流程 ----------
    def dispatch(self, context_id: str, text: str, file_bytes: bytes | None = None):
        pages = self._pdf_pages(file_bytes) if file_bytes else []
        page_map = dict(pages)
        plan = self._plan_with_gpt(text, [pid for pid, _ in pages])

        for agent, pid_list in plan.items():
            snap = self.memory.get(context_id)
            for pid in pid_list:
                prompt = (
                    "你是一名 GAIA benchmark 专家。请根据下列任务指令和对应 PDF 页面内容回答：\n"
                    f"【任务指令】{text}\n\n【PDF {pid} 内容】\n{page_map.get(pid, '')}"
                )
                payload = {
                    "prompt": prompt,
                    "context_id": context_id,
                    "page_id": pid,
                    "agent_name": agent,
                    "shared_memory": snap
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

        # ---------- 总结 ----------
        summary_prompt = "请基于各 agent 输出撰写 GAIA 风格总结（共识/差异/建议）：\n\n"
        for ag in plan:
            summary_prompt += f"【{ag}】{self.memory.get(context_id).get(ag, '')}\n\n"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}])
        summary = res.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append({"agent": "scheduler", "subtask": "自动生成总结", "output": summary})

        return self.memory.get(context_id), self.trace

