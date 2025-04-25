from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from inference import predict_audio_deepfake
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return JSONResponse(content={"status": "OK",
                                 "message": "Audio deepfake detection API is running."})

@app.get("/health")
def health_check():
    return JSONResponse(content={"health": "ok"})

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction = predict_audio_deepfake(file_path)
        if prediction is not None:
            return JSONResponse(content={"prediction": prediction})
        else:
            raise HTTPException(status_code=500, detail="Prediction failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8980, reload=True)