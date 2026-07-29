# import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from torchvision import models, transforms
import torch
import torch.nn as nn
import json
from pathlib import Path
import traceback

app = FastAPI(
    title="model mobilenetv2 lagi di load.. untuk clasification api",
    description="API integrasi ke web dari model mobilentev2 pytorch",
    version="1.0.0"
)

# konfigurasi perangkat

base_dir = Path(__file__).resolve().parent
class_names = base_dir / "model" / "class_names.json"
model_path = base_dir /  "model" / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

jumlah_kelas = 10



try:
    with open(class_names, "r") as f:
        class_names = json.load(f)
except Exception as e:
    print(f"gagal meagmbil data json{e}")
    class_names = [f"kelas {i}" for i in range(jumlah_kelas)]

def load_model(weights_pth :str):
    model = models.mobilenet_v2(weights=None) # Inisialisasi arsitektur MobileNetV2 tanpa pretrained weights bawaan

    # Sesuaikan layer classifier terakhir dengan jumlah kelas kategori yang kita punya
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, jumlah_kelas)
    #load bobot daro file pth
    state_dict = torch.load(weights_pth, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval() # kita masuk ke mode evak
    return model

try:
    model = load_model(model_path)
except Exception as e :
    raise RuntimeError(f"gagal laod model{str(e)}")
    

# Pipeline Preprocessing Gambar (ImageNet Standard)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

#endpoint FastAPI
@app.get("/")
def index():
    return{"status": "ok", "meessage":"mobilenetv2 api berjalan"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="error, karena file gambar yang di upload bukan jpg/png"
        )
    # baca file
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Transformasi input gambar
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilitas = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilitas, 0)

        idx = predicted_idx.item()
        label = class_names[idx] if idx < len(class_names) else str(idx)
        return{
            "success":True,
            "prediction": {
                "class_id":idx,
                "label":label,
                "confidence":round(confidence.item(), 4)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"terjadi kesalahan,file tidak dapap dibaca ada kesalahan saat memproses gambar: {str(e)}"
        )
