from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from infer import run_inference

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/infer")
async def infer(input_data: TextInput):
    result = run_inference(input_data.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)