import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CropYieldDataset
from src.predict import predict_mlp

BATCH_SIZE = 32
N_EPOCHS = 200


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        return x + self.positional_embedding


class CropTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                 dropout=0.1):
        super(CropTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        out = self.fc_out(x)
        return out


def train_transformer_model(data_df, train_percentage, yield_scaler, seq_len=3,
                            d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                            learning_rate=0.001, dropout=0.1):
    train_dataset = CropYieldDataset(data_df, is_train=True, seq_len=seq_len, train_percentage=train_percentage)
    valid_dataset = CropYieldDataset(data_df, is_train=False, seq_len=seq_len, train_percentage=train_percentage)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_dataset[0][0].shape[1]
    output_dim = train_dataset[0][1].shape[0]

    model = CropTransformerModel(input_dim, output_dim, seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                 dim_feedforward=dim_feedforward, dropout=dropout)

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
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)

        epoch_valid_loss = 0.0
        model.eval()
        for x_batch, y_batch in valid_dataloader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_valid_loss += loss.item()

        epoch_valid_loss /= len(valid_dataloader)

        print(f"Epoch {epoch}, Train loss: {epoch_train_loss:.4f}, Valid loss: {epoch_valid_loss:.4f}")

        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}, best loss {best_val_loss:.4f}")
                break

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)

    model.load_state_dict(best_model_state)

    predictions, pred_loss = predict_mlp(model, valid_dataloader, yield_scaler, data_df, seq_len, train_percentage, criterion)

    return model, predictions, train_losses, valid_losses, pred_loss
