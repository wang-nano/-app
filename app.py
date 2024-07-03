from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import torch
from torchvision import transforms
import json
from io import BytesIO
from model import swin_base_patch4_window7_224 as create_model
from plant_name import name_list
from plant_data import plant_inf

app = FastAPI()

# 加载类索引
json_path = './class_indices.json'
with open(json_path, "r") as json_file:
    class_indict = json.load(json_file)

# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=67).to(device)
model_weight_path = "model-90.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

data_transform = transforms.Compose(
    [transforms.Resize(int(224 * 1.143)),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        img = data_transform(image)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # 获取植物名称和信息
        plant_name = name_list[predict_cla]
        plant_info = plant_inf[plant_name]

        result = {
            "class": class_indict[str(predict_cla)],
            "probability": float(predict[predict_cla].numpy()),
            "plant_name": plant_name,
            "plant_info": plant_info
        }
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
