import os
import io
import requests
from typing import List, Tuple, Dict

import PyPDF2
from openai import OpenAI

from mcp.memory import MemoryStore

# ==================== 配置 ====================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, str] = {
    "llama2_agent": "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}

CHUNK_SIZE = 1200  # 约 ~400 token

# ==================== Scheduler ====================
class LLMScheduler:
    """调度流程：
    1. 接收用户任务文本 + 可选 PDF
    2. PDF → 纯文本 → 切分为 group_n 片段
    3. 把『完整任务文本』+『group ID 列表』交 GPT‑4o，让其判断并生成分配方案
    4. 根据方案把 (instruction+context) 合并为 prompt，发送到本地 /infer
    5. 汇总结果并再请 GPT‑4o 生成总结
    """

    def __init__(self):
        self.memory = MemoryStore()
        self.trace = []

    # ---------- 工具 ----------
    @staticmethod
    def _pdf_to_text(pdf_bytes: bytes) -> str:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)

    @staticmethod
    def _split_text(text: str) -> List[Tuple[str, str]]:
        chunks, idx, buf = [], 1, []
        for para in text.splitlines():
            if len("\n".join(buf) + para) > CHUNK_SIZE:
                chunks.append((f"group_{idx}", "\n".join(buf).strip()))
                idx += 1
                buf = [para]
            else:
                buf.append(para)
        if buf:
            chunks.append((f"group_{idx}", "\n".join(buf).strip()))
        return chunks

    # ---------- 让 GPT‑4o 做任务判断 + 分配 ----------
    def _plan_with_gpt(self, user_text: str, groups: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        group_ids = ", ".join(cid for cid, _ in groups) if groups else "(无)"
        sys_prompt = (
            "你是任务调度专家。\n"
            "给定完整的用户任务文本，以及文件材料的若干 group ID 列表（不含正文）。\n"
            "若材料为空，则只基于用户任务；否则请判断材料是否需要处理，并将需要处理的 group 分配给下列 agent：\n"
            + ", ".join(REGISTRY.keys()) + "\n"
            "输出格式：agent_name: id1,id2 或 agent_name: start-end。除分配结果外不要解释。"
        )
        user_prompt = (
            f"【用户任务全文】\n{user_text}\n\n"
            f"【文件材料 group ID 列表】{group_ids}"
        )
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
                    prefix = ''.join(filter(str.isalpha, start)) or 'group_'
                    s = int(''.join(filter(str.isdigit, start)))
                    e = int(''.join(filter(str.isdigit, end)))
                    plan[agent] += [f"{prefix}{i}" for i in range(s, e + 1)]
                else:
                    plan[agent].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- 主入口 ----------
    def dispatch(self, context_id: str, text: str, file_bytes: bytes | None = None):
        groups: List[Tuple[str, str]] = []
        if file_bytes:
            pdf_text = self._pdf_to_text(file_bytes)
            groups = self._split_text(pdf_text)

        plan = self._plan_with_gpt(text, groups)
        id2text = dict(groups)

        for agent, gid_list in plan.items():
            snap = self.memory.get(context_id)
            for gid in gid_list:
                context_text = id2text.get(gid, "")
                prompt = f"{text}\n\n{context_text}".strip()
                payload = {
                    "prompt": prompt,            # 兼容现有 /infer 接口
                    "context_id": context_id,
                    "chunk_id": gid,
                    "shared_memory": snap
                }
                try:
                    resp = requests.post(REGISTRY[agent], json=payload, timeout=60)
                    out = resp.json()
                    mem = out.get("memory_update") or {agent: out.get("result", "")}
                except Exception as e:
                    self.trace.append({"agent": agent, "subtask": gid, "output": f"[❌ 调用失败] {e}"})
                    continue

                self.memory.update(context_id, mem)
                self.trace.append({"agent": agent, "subtask": gid, "output": mem.get(agent, "")})

        # -------- GPT‑4o 总结 --------
        summary_prompt = "请根据以下结果撰写总结，包含共识、差异和建议：\n\n"
        for agent in plan:
            summary_prompt += f"【{agent}】{self.memory.get(context_id).get(agent, '')}\n\n"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}])
        summary = res.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append({"agent": "scheduler", "subtask": "自动生成总结", "output": summary})

        return self.memory.get(context_id), self.trace

