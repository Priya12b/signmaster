import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ml_pipeline.model import get_asl_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_LOADED_MODEL = None
_LOADED_CLASSES = None
_IMG_SIZE = 128
_DEVICE = None

def load_predictor(model_path=None):
    global _LOADED_MODEL, _LOADED_CLASSES, _IMG_SIZE, _DEVICE

    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_CLASSES, _DEVICE

    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "asl_mobilenetv2_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=_DEVICE)

    _LOADED_CLASSES = checkpoint.get("class_names")
    architecture = checkpoint.get("architecture", "mobilenet_v2")
    _IMG_SIZE = checkpoint.get("img_size", 128)

    model = get_asl_model(num_classes=len(_LOADED_CLASSES), pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_DEVICE)
    model.eval()

    _LOADED_MODEL = model
    return _LOADED_MODEL, _LOADED_CLASSES, _DEVICE

def get_inference_transform(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_image(image_input, top_k=3):
    """
    Predicts ASL alphabet from image input.
    image_input can be: file path (str), PIL.Image, or numpy ndarray (e.g. cv2 BGR frame).
    Returns dict with predicted_class, confidence, and top_k alternatives.
    """
    model, class_names, device = load_predictor()

    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif hasattr(image_input, "shape"): # numpy array from cv2
        import cv2
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            rgb_arr = image_input
        pil_img = Image.fromarray(rgb_arr)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type.")

    tf = get_inference_transform(_IMG_SIZE)
    tensor = tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))

    top_results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        top_results.append({
            "class": class_names[idx],
            "confidence": round(prob, 4)
        })

    best_class = top_results[0]["class"]
    best_conf = top_results[0]["confidence"]

    return {
        "predicted_class": best_class,
        "confidence": best_conf,
        "top_predictions": top_results
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = predict_image(sys.argv[1])
        print(f"Prediction: {res['predicted_class']} ({res['confidence'] * 100:.2f}%)")
        print(f"Top 3: {res['top_predictions']}")
    else:
        print("Usage: python -m ml_pipeline.predict <path_to_image>")
