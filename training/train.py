import torch
import torch.nn as nn
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from training.dataset import TRAINING_DATA
from app.classifier import CalorieClassifier

# Prepare data
texts = [item[0] for item in TRAINING_DATA]
labels = [item[1] for item in TRAINING_DATA]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)
X = vectorizer.fit_transform(texts).toarray()
y = labels

# Get actual input size from data
input_size = X.shape[1]
print(f"Vocabulary size: {input_size}")

# Train/val/test split 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model with actual input size
model = CalorieClassifier(input_size=input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Training...")
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val_t).float().mean()
            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2%}")

# Test evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, test_predicted = torch.max(test_outputs, 1)
    test_acc = (test_predicted == y_test_t).float().mean()
    print(f"\nTest Accuracy: {test_acc:.2%}")

# Save model with input size and vectorizer
os.makedirs("models", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size
}, "models/classifier.pt")

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved to models/classifier.pt")
print("Vectorizer saved to models/vectorizer.pkl")