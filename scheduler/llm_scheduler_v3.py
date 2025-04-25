import os
import requests
from openai import OpenAI
from mcp.memory import MemoryStore
import PyPDF2
import io

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY = {
    "llama2_agent": "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}

class LLMScheduler:
    def __init__(self):
        self.memory = MemoryStore()
        self.trace = []

    def split_pdf(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        pages = []
        for i, page_obj in enumerate(pdf_reader.pages):
            text = page_obj.extract_text()
            pages.append((i + 1, text))
        return pages

    def plan(self, user_task, is_pdf=False, pdf_file=None):
        if is_pdf:
            pdf_pages = self.split_pdf(pdf_file)
            total_pages = len(pdf_pages)
            messages = [
                {"role": "system", "content": (
                    "你是一个任务调度专家，请根据任务内容，将PDF页面拆解为子任务，并分配给 agent：\n"
                    f"任务描述：{user_task}\nPDF 页面数：{total_pages}\n"
                    "格式：agent_name: start-end"
                )},
                {"role": "user", "content": f"任务描述：{user_task}"}
            ]
            res = client.chat.completions.create(model="gpt-4o", messages=messages)
            return self.parse_task_assignment(res.choices[0].message.content)
        else:
            messages = [
                {"role":"system","content":(
                    "你是一个任务调度专家，请拆解任务并分配给 agent：\n"
                    f"任务：{user_task}" )},
                {"role":"user","content":f"任务：{user_task}"}
            ]
            res = client.chat.completions.create(model="gpt-4o", messages=messages)
            lines = res.choices[0].message.content.splitlines()
            return {l.split(":")[0].strip(): l.split(":")[1].strip() for l in lines if l.split(":")[0].strip() in REGISTRY}

    def parse_task_assignment(self, text):
        plan = {}
        for line in text.splitlines():
            if ':' in line:
                agent, rng = line.split(':')
                plan[agent.strip()] = rng.strip()
        return plan

    def dispatch(self, context_id, task, is_pdf=False, pdf_file=None):
        plan = self.plan(task, is_pdf, pdf_file)
        for agent, rng in plan.items():
            memory = self.memory.get(context_id)
            start, end = map(int, rng.split('-'))
            pages = self.split_pdf(pdf_file)[start-1:end]
            for idx, content in pages:
                try:
                    res = requests.post(REGISTRY[agent], json={
                        "context_id": context_id,
                        "agent_name": agent,
                        "subtask": content,
                        "shared_memory": memory
                    })
                    output = res.json()
                    if "memory_update" in output:
                        mem = output["memory_update"]
                    else:
                        mem = {agent: output.get("result", "")}
                except Exception as e:
                    self.trace.append({"agent":agent, "subtask":content, "output":f"[❌ 调用失败] {e}"})
                    continue

                self.memory.update(context_id, mem)
                self.trace.append({
                    "agent": agent,
                    "subtask": content,
                    "output": mem.get(agent, "")
                })

        # 自动总结
        prompt = "请根据以下结果撰写总结：\n"
        for agent in plan:
            prompt += f"【{agent}】：{self.memory.get(context_id).get(agent, '')}\n"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        summary = res.choices[0].message.content
        self.memory.update(context_id, {"summary":summary})
        self.trace.append({"agent":"scheduler","subtask":"自动生成总结","output":summary})
        return self.memory.get(context_id), self.trace

