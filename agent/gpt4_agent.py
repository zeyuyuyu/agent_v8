from openai import OpenAI
import os

# 初始化 GPT-4 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def handle(subtask, shared_memory):
    """
    处理子任务，调用 GPT-4 模型
    :param subtask: 子任务的文本内容
    :param shared_memory: 共享内存
    :return: 结果和更新后的内存
    """
    # 发送请求给 GPT-4 模型
    messages = [
        {"role": "system", "content": "你是一个擅长分析的 GPT Agent"},
        {"role": "user", "content": subtask}
    ]

    res = client.chat.completions.create(
        model="gpt-4",  # 使用 GPT-4 模型
        messages=messages
    )

    # 获取 GPT-4 的返回结果
    result = res.choices[0].message['content'].strip()

    # 返回结果和更新的内存（此处假设内存没有做特殊修改）
    return result, shared_memory

