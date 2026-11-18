import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ml_pipeline.model import get_asl_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def evaluate_model(model_path=None, test_dir=None, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "asl_mobilenetv2_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names")
    architecture = checkpoint.get("architecture", "mobilenet_v2")
    img_size = checkpoint.get("img_size", 128)

    model = get_asl_model(num_classes=len(class_names), pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if test_dir is None:
        test_dir = os.path.join(DATA_DIR, "test")

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    print(f"Evaluating model on {len(test_dataset)} test samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)

    print("\n================ TEST EVALUATION RESULTS ================")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report_text)

    # Save reports
    with open(os.path.join(MODELS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report_text)

    eval_summary = {
        "test_accuracy": round(float(test_acc), 4),
        "test_samples": len(test_dataset),
        "num_classes": len(class_names),
        "architecture": architecture,
        "classes": class_names,
        "metrics": {
            "macro_avg_precision": round(report_dict["macro avg"]["precision"], 4),
            "macro_avg_recall": round(report_dict["macro avg"]["recall"], 4),
            "macro_avg_f1_score": round(report_dict["macro avg"]["f1-score"], 4),
            "weighted_avg_f1_score": round(report_dict["weighted avg"]["f1-score"], 4)
        }
    }

    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(eval_summary, f, indent=4)

    # Plot and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"ASL Alphabet Confusion Matrix (Test Accuracy: {test_acc * 100:.2f}%)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Format numbers inside matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return eval_summary

if __name__ == "__main__":
    evaluate_model()
