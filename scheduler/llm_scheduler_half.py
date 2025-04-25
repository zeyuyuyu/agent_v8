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

    # PDF拆分函数
    def split_pdf(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        pages = []
        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i].extract_text()
            pages.append((i + 1, page))  # 返回页号和页内容
        return pages

    def plan(self, user_task, is_pdf=False, pdf_file=None):
        if is_pdf:
            # 如果是PDF任务，拆分并生成子任务
            pdf_pages = self.split_pdf(pdf_file)
            total_pages = len(pdf_pages)
            
            # 创建一个更加明确的任务分配提示，让GPT-4o清晰地分配页面给代理
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个任务调度专家，请根据任务内容，将PDF页面拆解为子任务，并分别分配给以下 agent 中的一个：\n"
                        "- llama2_agent\n- llama2_agent_2\n"
                        "请根据以下信息，生成合理的任务分配，明确指出每个代理负责哪些页面：\n"
                        f"任务描述：{user_task}\n"
                        f"PDF页面数：{total_pages}\n"
                        "请明确列出每个代理负责的页面范围，并且确保合理分配，格式如下：\n"
                        "代理名称: 页面范围（例如：llama2_agent: 1-3，llama2_agent_2: 4-5）"
                    )
                },
                {"role": "user", "content": f"任务描述：{user_task}"}
            ]
            
            # 请求 GPT-4o 来分配页面
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            task_assignment = res.choices[0].message.content.strip()
            
            # 返回 GPT-4o 给出的任务分配结果
            return self.parse_task_assignment(task_assignment)
        else:
            # 如果是文本任务
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个任务调度专家，请根据任务内容，将其拆解为子任务，并分别分配给以下 agent 中的一个：\n"
                        "- llama2_agent\n- llama2_agent_2\n"
                        "请根据以下信息，生成合理的任务分配：\n"
                        f"任务描述：{user_task}\n"
                    )
                },
                {"role": "user", "content": f"任务：{user_task}"}
            ]
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            lines = res.choices[0].message.content.strip().splitlines()
            return {
                l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip()
                for l in lines if ":" in l and l.split(":")[0].strip() in REGISTRY
            }

    def parse_task_assignment(self, task_assignment):
        """
        将 GPT-4o 返回的任务分配字符串解析为代理和页面的映射
        示例输入:
            llama2_agent: 1-3
            mistral_agent: 4-5
        """
        plan = {}
        lines = task_assignment.splitlines()
        for line in lines:
            if ":" in line:
                agent, pages = line.split(":")
                agent = agent.strip()
                pages = pages.strip()
                plan[agent] = pages
        return plan

    def dispatch(self, context_id, task, is_pdf=False, pdf_file=None):
        plan = self.plan(task, is_pdf, pdf_file)

        for agent, page_range in plan.items():
            memory = self.memory.get(context_id)

            # 解析页面范围
            page_start, page_end = map(int, page_range.split('-'))
            # 获取指定范围的PDF页面
            pdf_pages = self.split_pdf(pdf_file)
            relevant_pages = pdf_pages[page_start - 1:page_end]

            # 发送任务给相应代理
            for subtask in relevant_pages:
                try:
                    res = requests.post(REGISTRY[agent], json={
                        "context_id": context_id,
                        "agent_name": agent,
                        "subtask": subtask[1],  # 提交每一页的内容
                        "shared_memory": memory
                    })
                    output = res.json()
                    if "memory_update" in output:
                         mem = output.get("memory_update", {})
                    else:
                        mem = {agent: output.get("result","")}
                except Exception as e:
                    mem = {}
                    self.trace.append({
                        "agent": agent,
                        "subtask": subtask[1],
                        "output": f"[❌ 调用失败] {str(e)}"
                    })
                    continue

                # 更新 memory + trace
                self.memory.update(context_id, mem)
                self.trace.append({
                    "agent": agent,
                    "subtask": subtask[1],
                    "output": mem.get(agent, "")
                })

        # ✅ 自动生成总结
        summary_prompt = "请根据以下多位智能体的分析结果，撰写一个总结报告，内容包括共识、差异、你的建议：\n\n"
        for agent in plan.keys():
            content = self.memory.get(context_id).get(agent, "")
            summary_prompt += f"【{agent}】：{content}\n\n"

        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = res.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append({
            "agent": "scheduler",
            "subtask": "自动生成总结",
            "output": summary
        })

        return self.memory.get(context_id), self.trace

