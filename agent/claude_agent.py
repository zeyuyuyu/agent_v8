import anthropic
import os

# 初始化 Claude 客户端
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def handle(subtask, shared_memory):
    """
    处理子任务，调用 Claude 模型
    :param subtask: 子任务的文本内容
    :param shared_memory: 共享内存
    :return: 结果和更新后的内存
    """
    # 调用 Claude 模型
    msg = client.messages.create(
        model="claude-3-opus-20240229",  # 使用 Claude 的模型
        max_tokens=1024,
        temperature=0.6,
        messages=[{"role": "user", "content": subtask}]
    )

    result = msg['text'].strip()  # 获取并处理 Claude 的返回结果

    # 返回结果和更新的内存（此处假设内存没有做特殊修改）
    return result, shared_memory

