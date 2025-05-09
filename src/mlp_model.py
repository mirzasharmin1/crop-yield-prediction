import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CropYieldDatasetSimple
from src.predict import predict_mlp

BATCH_SIZE = 32
N_EPOCHS = 200


class MLPModel(nn.Module):
    def __init__(self, num_layers, hidden_dim, input_size, output_size):
        super(MLPModel, self).__init__()

        layers = [
            nn.Linear(input_size, hidden_dim)
        ]

        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_model(data_df, train_percentage, yield_scaler, num_layers, hidden_dim, seq_len=3, learning_rate=0.001):
    train_dataset_simple = CropYieldDatasetSimple(
        data_df, is_train=True, seq_len=seq_len, train_percentage=train_percentage
    )
    valid_dataset_simple = CropYieldDatasetSimple(
        data_df, is_train=False, seq_len=seq_len, train_percentage=train_percentage
    )

    train_dataloader = DataLoader(train_dataset_simple, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset_simple, batch_size=BATCH_SIZE, shuffle=False)

    input_size = train_dataset_simple[0][0].shape[0]
    output_size = valid_dataset_simple[0][1].shape[0]

    model = MLPModel(num_layers=num_layers, hidden_dim=hidden_dim, input_size=input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    best_model_state = model.state_dict()

    for epoch in range(N_EPOCHS):
        epoch_train_loss = 0.0

        model.train()
        for x_batch, y_batch in train_dataloader:
            y_pred = model(x_batch)
            train_loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

        epoch_train_loss /= len(train_dataloader)

        epoch_valid_loss = 0.0

        model.eval()
        for x_batch, y_batch in valid_dataloader:
            y_pred = model(x_batch)
            valid_loss = criterion(y_pred, y_batch)

            epoch_valid_loss += valid_loss.item()

        epoch_valid_loss /= len(valid_dataloader)

        print(f"Epoch {epoch}, Train loss: {epoch_train_loss:.4f}, Valid loss: {epoch_valid_loss:.4f}")

        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1

            if counter >= patience:
                print(f"Early stopping triggered, best loss {best_val_loss:.4f}")
                break

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)

    model.load_state_dict(best_model_state)

    predictions, pred_loss = predict_mlp(model, valid_dataloader, yield_scaler, data_df, seq_len, train_percentage, criterion)

    return model, predictions, train_losses, valid_losses, pred_loss
