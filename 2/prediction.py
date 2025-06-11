import torch
from torchvision import transforms
from PIL import Image
from resnet34_manual import ResNet34

CHECKPOINT_PATH = 'final_model.pth'
INPUT_IMAGE = 'cat.jpg'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['Cat', 'Dog']

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def prepare_image(path):
    img = Image.open(path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor.to(DEVICE)

# --- Загрузка модели ---
def load_model(path):
    model = ResNet34(num_classes=2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# --- Предсказание ---
def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        return LABELS[predicted_idx]

if __name__ == "__main__":
    model = load_model(CHECKPOINT_PATH)
    image_tensor = prepare_image(INPUT_IMAGE)
    result = predict(model, image_tensor)
    print(f"Prediction: {result}")
