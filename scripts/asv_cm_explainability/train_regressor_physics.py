import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# Define dataset class
class ASVspoofDataset(Dataset):
    def __init__(self, embeddings_dir, metadata, label_col, dim_range=[0, 0]):
        self.embeddings_dir = embeddings_dir
        self.metadata = metadata
        self.label_col = label_col
        self.dim_range = dim_range

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        asvspoof_id = self.metadata.iloc[idx, 0]
        embedding_path = os.path.join(self.embeddings_dir, asvspoof_id + ".npy")
        embedding = np.load(embedding_path)
        if self.dim_range != [0, 0]:
            start_dim, end_dim = self.dim_range
            embedding = embedding[start_dim:end_dim]
        label = self.metadata.iloc[idx][self.label_col]

        src_data = torch.tensor(embedding, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        ).unsqueeze(0)
        return src_data


# Define model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


def load_metadata(filepath):
    full_metadata = pd.read_csv(filepath, sep="\t")
    metadata = full_metadata.dropna()
    return metadata


# Filter the DataFrame
def filter_metadata(metadata, embeddings_dir, partition):
    filtered_metadata = metadata[
        metadata["ASVSPOOF_ID"].apply(
            lambda x: os.path.exists(
                os.path.join(embeddings_dir, partition, x + ".npy")
            )
        )
    ]
    return filtered_metadata


def preprocess_metadata(metadata, embeddings_dir):
    # Partition the data according to ASVspoof convention
    train_metadata = filter_metadata(
        metadata[metadata["PARTITION"].str.contains("train")], embeddings_dir
    )
    eval_metadata = filter_metadata(
        metadata[metadata["PARTITION"].str.contains("eval")], embeddings_dir
    )
    return train_metadata, eval_metadata


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    # rmse = np.sqrt(mse + 1e-6)
    rmse = np.sqrt(((actuals - predictions) ** 2).mean())
    r2 = r2_score(actuals, predictions)
    return mae, mse, rmse, r2


def main(config_file):
    # load experiment configurations
    with open(config_file, "r") as f_json:
        config = json.loads(f_json.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_filepath = "data/Database/ASVspoof_VCTK_aligned_physical_meta.tsv"
    embeddings_dir = config["embeddings_dir"] + "/whole"
    label_cols = ["PITCH", "SPK_RATE", "DURATION", "SNR"]
    label_col = config["trait"]
    assert label_col in label_cols

    metadata = load_metadata(metadata_filepath)
    train_metadata, eval_metadata = preprocess_metadata(metadata, embeddings_dir)

    embedding_dim_range = config["input"]["dim_range"]
    train_dataset = ASVspoofDataset(
        embeddings_dir,
        train_metadata,
        label_col,
        dim_range=embedding_dim_range,
    )
    dev_dataset = ASVspoofDataset(
        embeddings_dir,
        train_metadata,
        label_col,
        dim_range=embedding_dim_range,
    )
    eval_dataset = ASVspoofDataset(
        embeddings_dir,
        eval_metadata,
        label_col,
        dim_range=embedding_dim_range,
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=config["model"]["input_dim"]).to(device)
    criterion = RMSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["adam_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["step_size"],
        gamma=config["training"]["gamma"],
    )

    best_rmse = float("inf")
    patience = 5
    no_improvement = 0

    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):  # Start with a larger number of epochs
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        mae, mse, rmse, r2 = evaluate_model(model, dev_loader, device)
        print(
            f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Dev RMSE: {rmse:.4f}, R²: {r2:.4f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            no_improvement = 0
        else:
            no_improvement += 1

        # if no_improvement >= patience:
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break

        scheduler.step()

    mae, mse, rmse, r2 = evaluate_model(model, eval_loader, device)
    print(
        f"Evaluation Set - MAE: {round(mae, 4)}, MSE: {round(mse, 4)}, RMSE: {round(rmse, 4)}, R²: {round(r2, 4)}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
