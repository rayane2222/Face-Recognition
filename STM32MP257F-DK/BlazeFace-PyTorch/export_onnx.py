import torch
from blazeface import BlazeFace

# Charger le modèle
model = BlazeFace()
model.load_weights("blazeface.pth")
model.eval()

# Dummy input (BlazeFace attend 128x128)
dummy_input = torch.randn(1, 3, 128, 128)

# Export ONNX
torch.onnx.export(
    model,
    dummy_input,
    "blazeface.onnx",
    input_names=["input"],
    output_names=["scores", "boxes"],
    opset_version=11
)

print("Export ONNX terminé : blazeface.onnx")
