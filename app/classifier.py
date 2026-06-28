import torch
import torch.nn as nn
import pickle
import os

CALORIE_RANGES = {
    0: "low",
    1: "medium",
    2: "high"
}

MODEL_PATH = "models/classifier.pt"
VECTORIZER_PATH = "models/vectorizer.pkl"


class CalorieClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


_model = None
_vectorizer = None


def load_model():
    global _model, _vectorizer

    if not os.path.exists(MODEL_PATH):
        print("No model found — classifier will be unavailable")
        return False

    if not os.path.exists(VECTORIZER_PATH):
        print("No vectorizer found — classifier will be unavailable")
        return False

    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    input_size = checkpoint['input_size']

    _model = CalorieClassifier(input_size=input_size)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()

    print(f"Classifier loaded — input size: {input_size}")
    return True


def predict_calorie_range(meal_description: str) -> dict:
    if _model is None or _vectorizer is None:
        return {
            "predicted_range": "unknown",
            "confidence_score": 0.0,
            "available": False
        }

    X = _vectorizer.transform([meal_description]).toarray()
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        outputs = _model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "predicted_range": CALORIE_RANGES[predicted_class],
        "confidence_score": round(confidence, 3),
        "available": True
    }