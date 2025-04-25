from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional
from scheduler.llm_scheduler import LLMScheduler

app = FastAPI()
scheduler = LLMScheduler()

@app.post("/submit_task")
async def submit_task(
    text: str = Form(...),                       # 用户任务全文
    file: Optional[UploadFile] = File(None)      # 可选 PDF 文件
):
    # 1️⃣ 读取 PDF（二进制）或置 None
    pdf_bytes = await file.read() if file else None

    # 2️⃣ 调用调度器：context_id 可自行生成，这里演示用固定值
    result, trace = scheduler.dispatch(
        context_id="ctx001",
        text=text,
        file_bytes=pdf_bytes
    )

    return {"memory": result, "trace": trace}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

