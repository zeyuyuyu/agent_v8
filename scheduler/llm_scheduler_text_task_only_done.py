import os
import io
import requests
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
from openai import OpenAI
from mcp.memory import MemoryStore

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------- 配置 ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, str] = {
    "llama2_agent": "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}

# =================== 调度器 ===================
class LLMScheduler:
    """1. 支持 PDF（二进制）→ 每页拆分
       2. 支持纯文本输入（plain_text）
       3. 仅有 task 指令时：用 GPT‑4o 将 task 拆成子任务并分配给 agent"""

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict] = []

    # ---------- PDF to pages ----------
    @staticmethod
    def _pdf_pages(data: bytes) -> List[Tuple[str, str]]:
        pages = []
        doc = fitz.open(stream=data, filetype="pdf")
        for idx, page in enumerate(doc, 1):
            text = page.get_text("text") or ""
            if not text.strip() and OCR_AVAILABLE:
                img = convert_from_bytes(data, dpi=300, first_page=idx, last_page=idx)[0]
                text = pytesseract.image_to_string(img, lang="eng+chi_sim")
            pages.append((f"page_{idx}", text.strip()))
        return pages

    # ---------- GPT‑4o 规划 ----------
    def _plan_pages(self, task: str, page_ids: List[str]) -> Dict[str, List[str]]:
        # 分配页面给 agent
        ids_str = ", ".join(page_ids)
        sys = ("你是任务调度专家，请根据【任务】分配页面给以下 agent：" + ", ".join(REGISTRY.keys()) +
               "。只返回分配结果，每行格式：agent: id1,id2 或 agent: m-n。")
        user = f"【任务】{task}\n【页面列表】{ids_str}"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":sys},{"role":"user","content":user}])
        return self._parse_plan(res.choices[0].message.content)

    def _plan_subtasks(self, task: str) -> Dict[str, List[str]]:
        # 将任务拆分为子任务并分配
        sys = ("你是任务调度专家，请将【任务】拆成若干可并行的子任务，并分配给以下 agent：" + ", ".join(REGISTRY.keys()) +
               "。输出格式：agent: 子任务描述1；agent: 子任务描述2,… 不要添加多余解释。")
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":sys},{"role":"user","content":task}])
        mapping: Dict[str, List[str]] = {k: [] for k in REGISTRY}
        for line in res.choices[0].message.content.splitlines():
            if ':' not in line:
                continue
            ag, desc = line.split(':', 1)
            ag = ag.strip()
            if ag in REGISTRY:
                mapping[ag].append(desc.strip())
        return {k: v for k, v in mapping.items() if v}

    @staticmethod
    def _parse_plan(text: str) -> Dict[str, List[str]]:
        plan: Dict[str, List[str]] = {k: [] for k in REGISTRY}
        for ln in text.splitlines():
            if ':' not in ln:
                continue
            ag, ids = ln.split(':', 1)
            ag = ag.strip()
            if ag not in REGISTRY:
                continue
            ids = ids.replace(' ', '')
            for part in ids.split(','):
                if not part:
                    continue
                if '-' in part:
                    s, e = part.split('-')
                    s_idx = int(''.join(filter(str.isdigit, s)))
                    e_idx = int(''.join(filter(str.isdigit, e)))
                    prefix = ''.join(filter(str.isalpha, s)) or 'page_'
                    plan[ag] += [f"{prefix}{i}" for i in range(s_idx, e_idx + 1)]
                else:
                    plan[ag].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- dispatch ----------
    def dispatch(
        self,
        context_id: str,
        task: str,
        pdf_bytes: Optional[bytes] = None,
        file_bytes: Optional[bytes] = None,
        plain_text: Optional[str] = None,
    ):
        # 1) 数据来源判断
        pdf_data = pdf_bytes or file_bytes
        if pdf_data:
            pages = self._pdf_pages(pdf_data)
            page_map = dict(pages)
            plan = self._plan_pages(task, [pid for pid, _ in pages])
            # prompt 生成函数
            def make_prompt(pid):
                return f"【任务指令】{task}\n\n【{pid} 原文】\n{page_map.get(pid,'')}"
        elif plain_text is not None:
            # 单段文本 -> 默认 agent0
            pages = [("text_1", plain_text.strip())]
            plan = {list(REGISTRY.keys())[0]: ["text_1"]}
            def make_prompt(_):
                return f"【任务指令】{task}\n\n【文本内容】\n{plain_text.strip()}"
        else:
            # 仅指令 -> GPT 规划子任务
            pages = []  # 不使用页面概念
            plan = self._plan_subtasks(task)
            def make_prompt(subtask_desc):
                return f"【总任务】{task}\n\n【子任务描述】{subtask_desc}"
        # 2) 分发
        for ag, keys in plan.items():
            snapshot = self.memory.get(context_id)
            for key in keys:
                prompt_text = make_prompt(key)
                payload = {
                    "prompt": prompt_text,
                    "context_id": context_id,
                    "sub_id": key,
                    "agent_name": ag,
                    "shared_memory": snapshot
                }
                try:
                    res = requests.post(REGISTRY[ag], json=payload, timeout=120)
                    data = res.json()
                    res_text = data.get('result', '')
                except Exception as e:
                    self.trace.append({"agent": ag, "subtask": key, "output": f"[❌ 调用失败] {e}"})
                    continue
                self.memory.update(context_id, {ag: {key: res_text}})
                self.trace.append({"agent": ag, "subtask": key, "output": res_text})
        # 3) 总结
        collected = "\n\n".join(f"【{ag}】\n" + "\n".join(v.values()) for ag, v in self.memory.get(context_id).items())
        rsp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":f"请综合各智能体输出，总结共识、差异并给建议：\n{collected}"}])
        summary = rsp.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append({"agent": "scheduler", "subtask": "自动总结", "output": summary})
        return self.memory.get(context_id), self.trace

