import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

def efficientnet_b0():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    return model

model = efficientnet_b0()
model.load_state_dict(torch.load("EFFICIENTNET_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
st.title("Aerial Object Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        conf = prob*100 if prob >= 0.5 else (1-prob)*100
    pred_label ="Drone" if prob > 0.5 else "Bird"

    st.write(f"Predicted Label: {pred_label}")
    st.write(f"Confidence: {conf:.2f}")
else:
    st.warning("Please upload an image file (jpg, jpeg, png) to get prediction.")
