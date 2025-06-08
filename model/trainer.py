import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 🔸 1. 데이터셋 정의
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir):
        self.X = []
        self.y = []
        self.labels = []

        for file in os.listdir(data_dir):
            if file.endswith(".npy"):
                label = file.split("_")[0]
                self.X.append(np.load(os.path.join(data_dir, file)))
                self.labels.append(label)

        self.encoder = LabelEncoder()
        self.y = self.encoder.fit_transform(self.labels)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

    def get_num_classes(self):
        return len(self.encoder.classes_)

    def get_label_encoder(self):
        return self.encoder

# 🔸 2. LSTM 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=8):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn.squeeze(0))
        return out

# 🔸 3. 학습 함수
def train_model(data_dir, model_path="model/model.pth", epochs=20, batch_size=8):
    dataset = SignLanguageDataset(data_dir)
    num_classes = dataset.get_num_classes()
    label_encoder = dataset.get_label_encoder()

    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.y, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = LSTMClassifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_encoder": label_encoder
    }, model_path)
    print(f"[INFO] 학습 완료 - 모델 저장됨: {model_path}")

# 🔸 4. 검증 함수
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return 100 * correct / total

# 🔸 5. 실행 구문
if __name__ == "__main__":
    train_model("data")