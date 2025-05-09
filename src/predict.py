import numpy as np
import pandas as pd

from src.dataset import get_countries_and_years
from src.preprocess import unscale_data


def predict_mlp(model, valid_dataloader, yield_scaler, data_df, seq_len, train_percentage, criterion):
    model.eval()
    predictions = []

    total_loss = 0.0

    for x_batch, y_batch in valid_dataloader:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        predictions.append(y_pred.detach().numpy())
        total_loss += loss.item()

    valid_loss = total_loss / len(valid_dataloader)

    predictions_array = np.vstack(predictions)
    unscaled = unscale_data(predictions_array, yield_scaler)
    unscaled = np.clip(unscaled, 0, None)

    countries_and_years = get_countries_and_years(data_df, seq_len, train_percentage)
    yield_cols = [col for col in data_df.columns if col.endswith('_Yield')]

    result_df = pd.DataFrame(countries_and_years, columns=['Country', 'Year'])
    yield_df = pd.DataFrame(unscaled, columns=yield_cols)

    final_df = pd.concat([result_df, yield_df], axis=1)
    return final_df, valid_loss
