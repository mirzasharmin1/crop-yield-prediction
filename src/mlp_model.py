import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CropYieldDatasetSimple, get_countries_and_years
from src.preprocess import unscale_data

BATCH_SIZE = 32
N_EPOCHS = 200


class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)


def predict_mlp(model, valid_dataloader, scaler, data_df, seq_len, train_percentage):
    model.eval()
    predictions = []

    for x_batch, _ in valid_dataloader:
        y_pred = model(x_batch)
        predictions.append(y_pred.detach().numpy())

    predictions_array = np.vstack(predictions)
    unscaled = unscale_data(predictions_array, scaler)
    unscaled = np.clip(unscaled, 0, None)

    countries_and_years = get_countries_and_years(data_df, seq_len, train_percentage)
    yield_cols = [col for col in data_df.columns if col.endswith('_Yield')]

    result_df = pd.DataFrame(countries_and_years, columns=['Country', 'Year'])
    yield_df = pd.DataFrame(unscaled, columns=yield_cols)

    final_df = pd.concat([result_df, yield_df], axis=1)
    return final_df


def train_mlp_model(data_df, train_percentage, scaler, seq_len=3, learning_rate=0.001):
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

    model = MLPModel(input_size=input_size, output_size=output_size)
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

        print(f"Epoch {epoch}, Train loss: {epoch_train_loss}, Valid loss: {epoch_valid_loss}")

        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1

            if counter >= patience:
                print(f"Early stopping triggered, best loss {best_val_loss}")
                break

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)

    model.load_state_dict(best_model_state)

    predictions = predict_mlp(model, valid_dataloader, scaler, data_df, seq_len, train_percentage)

    return model, predictions, train_losses, valid_losses
