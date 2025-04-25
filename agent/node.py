import sys
from fastapi import FastAPI, Request
import uvicorn

# 获取命令行参数，传入的代理名称和端口号
agent_name = sys.argv[1]
port = int(sys.argv[2])

# 动态导入相应的代理模块
agent = __import__(f"agent.{agent_name}", fromlist=["handle"])

# 创建 FastAPI 实例
app = FastAPI()

@app.post("/run")
async def run(request: Request):
    """
    接收任务请求并处理。
    :param request: 请求内容，包含任务信息和共享内存。
    :return: 任务处理结果和更新的内存信息。
    """
    data = await request.json()
    context_id = data["context_id"]
    subtask = data["subtask"]
    memory = data.get("shared_memory", {})

    # 调用动态导入的代理模块处理子任务
    result, updated_memory = agent.handle(subtask, memory)
    
    return {
        "result": result,
        "memory_update": {agent_name: updated_memory}
    }

if __name__ == "__main__":
    # 启动代理的 FastAPI 服务器
    uvicorn.run(app, host="0.0.0.0", port=port)

