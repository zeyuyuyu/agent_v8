import os
import io
import requests
from typing import List, Tuple, Dict

import PyPDF2  # 纯粹文本抽取即可满足需求
from openai import OpenAI

from mcp.memory import MemoryStore

# ==================== 初始化 ====================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 本地可调用的 LLM Agent 注册表（/infer 端点）
REGISTRY: Dict[str, str] = {
    "llama2_agent": "http://136.59.129.136:34517/infer",
    "mistral_agent": "http://136.59.129.136:34749/infer"
}

CHUNK_SIZE = 1200  # 每个文本块最大 tokens 左右（粗估字符）

# ==================== Scheduler 实现 ====================

class LLMScheduler:
    """负责：
    1. 读取 PDF -> 纯文本
    2. 按固定大小切分并编号
    3. 用 GPT‑4o 规划每个 Agent 处理哪些 chunk
    4. 分发子任务（instruction + context）到本地 Agent
    5. 汇总结果并生成总结
    """

    def __init__(self):
        self.memory = MemoryStore()
        self.trace = []

    # -------- PDF 读取 & 切分 --------
    def _pdf_to_text(self, pdf_binary: bytes) -> str:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_binary))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)

    def _split_text(self, text: str) -> List[Tuple[str, str]]:
        """把大文本按 CHUNK_SIZE 粗分，返回 [(chunk_id, chunk_text), ...]"""
        chunks, idx, buf = [], 1, []
        for para in text.splitlines():
            if len("\n".join(buf) + para) > CHUNK_SIZE:
                chunks.append((f"chunk_{idx}", "\n".join(buf).strip()))
                idx += 1
                buf = [para]
            else:
                buf.append(para)
        if buf:
            chunks.append((f"chunk_{idx}", "\n".join(buf).strip()))
        return chunks

    # -------- 任务规划 --------
    def _plan(self, user_task: str, chunks: List[Tuple[str, str]]):
        """调用 GPT‑4o，将 chunk ID 分给各 agent。
        GPT 只看到 chunk_id，避免暴露全文；同时指明要返回 "agent: id1,id2" 或 "agent: id_start-id_end"""        
        chunk_ids = ", ".join(cid for cid, _ in chunks)
        sys_prompt = (
            "你是任务调度专家。下面给出若干文本块 ID，请将它们合理分配给下列 agent，输出格式严格如下：\n"
            "agent_name: id1,id2 或 agent_name: start-end（闭区间）\n"
            "agent 必须只用以下名字：" + ", ".join(REGISTRY.keys()) + "。\n"
            "只返回分配结果，不要添加额外解释。"
        )
        user_prompt = (
            f"用户任务：{user_task}\n"
            f"待分配文本块：{chunk_ids}"
        )
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        )
        return self._parse_plan(res.choices[0].message.content)

    def _parse_plan(self, plan_text: str) -> Dict[str, List[str]]:
        plan: Dict[str, List[str]] = {k: [] for k in REGISTRY}
        for line in plan_text.splitlines():
            if ":" not in line:
                continue
            agent, ids = line.split(":", 1)
            agent = agent.strip()
            if agent not in REGISTRY:
                continue
            ids = ids.replace(" ", "")
            id_list = []
            for part in ids.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    prefix = ''.join(filter(str.isalpha, start)) or 'chunk_'
                    s_num = int(''.join(filter(str.isdigit, start)))
                    e_num = int(''.join(filter(str.isdigit, end)))
                    id_list += [f"{prefix}{i}" for i in range(s_num, e_num + 1)]
                else:
                    id_list.append(part)
            plan[agent] += id_list
        # 清理空 agent
        return {k: v for k, v in plan.items() if v}

    # -------- 分发执行 --------
    def dispatch(self, context_id: str, user_task: str, pdf_binary: bytes):
        full_text = self._pdf_to_text(pdf_binary)
        chunks = self._split_text(full_text)
        plan = self._plan(user_task, chunks)
        id2text = dict(chunks)

        for agent, id_list in plan.items():
            memory_snapshot = self.memory.get(context_id)
            for cid in id_list:
                context_text = id2text.get(cid, "")
                payload = {
                    "context_id": context_id,
                    "agent_name": agent,
                    "instruction": user_task,  # GPT‑4o 指令原文
                    "context": context_text,   # PDF 对应块内容
                    "chunk_id": cid,
                    "shared_memory": memory_snapshot
                }
                try:
                    resp = requests.post(REGISTRY[agent], json=payload, timeout=60)
                    out = resp.json()
                    # 兼容 result / memory_update
                    mem = out.get("memory_update") or {agent: out.get("result", "")}
                except Exception as e:
                    self.trace.append({"agent": agent, "subtask": cid, "output": f"[❌ 调用失败] {e}"})
                    continue

                self.memory.update(context_id, mem)
                self.trace.append({"agent": agent, "subtask": cid, "output": mem.get(agent, "")})

        # -------- 汇总总结 --------
        summary_prompt = "请根据以下多位智能体的分析结果，撰写总结，包含共识、差异和建议：\n\n"
        for agent in plan:
            summary_prompt += f"【{agent}】{self.memory.get(context_id).get(agent, '')}\n\n"
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = res.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append({"agent": "scheduler", "subtask": "自动生成总结", "output": summary})

        return self.memory.get(context_id), self.trace

