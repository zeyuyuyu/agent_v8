from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional
from scheduler.llm_scheduler import LLMScheduler

app = FastAPI()

scheduler = LLMScheduler()

@app.post("/submit_task")
async def submit_task(
    text: str = Form(...),  # 任务文本
    file: Optional[UploadFile] = File(None)  # 可选PDF文件
):
    if file:
        # 如果上传的是PDF文件
        pdf_file = await file.read()
        result, trace = scheduler.dispatch("context_id", text, is_pdf=True, pdf_file=pdf_file)
    else:
        # 如果只有文本
        result, trace = scheduler.dispatch("context_id", text)

    return {"result": result, "trace": trace}

