import random

import matplotlib.pyplot as plt


def plot_losses(train_losses, valid_losses, figname=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if figname is not None:
        plt.savefig(figname)

    plt.show()


def plot_crop_yield_predictions(predicted_df, true_df, min_data_points=5, figname=None):
    crop_names = [col.removesuffix("_Yield") for col in predicted_df.columns.tolist()[2:]]

    # Identify valid (country, crop) pairs based on data availability in true_df
    valid_pairs = []
    for country in true_df['Country'].unique():
        country_data = true_df[true_df['Country'] == country]
        for crop in crop_names:
            crop_col = f"{crop}_Yield"
            if crop_col in country_data.columns:
                num_valid = country_data[crop_col].notna().sum()
                if num_valid >= min_data_points:
                    valid_pairs.append((country, crop))

    if len(valid_pairs) < 4:
        raise ValueError("Not enough valid (country, crop) pairs with sufficient data.")

    selected_pairs = random.sample(valid_pairs, 4)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
    axs = axs.flatten()

    for i, (country, crop) in enumerate(selected_pairs):
        crop_col = f"{crop}_Yield"

        country_pred = predicted_df[predicted_df['Country'] == country]
        country_true = true_df[true_df['Country'] == country]

        merged = country_pred[['Year', crop_col]].merge(
            country_true[['Year', crop_col]],
            on='Year',
            suffixes=('_pred', '_true')
        ).dropna().sort_values('Year')

        axs[i].plot(merged['Year'], merged[f"{crop}_Yield_pred"], label='Predicted', marker='o')
        axs[i].plot(merged['Year'], merged[f"{crop}_Yield_true"], label='Actual', marker='x')
        axs[i].set_title(f"{country} - {crop}")
        axs[i].set_xlabel("Year")
        axs[i].set_ylabel("Yield")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()

    if figname is not None:
        plt.savefig(figname)

    plt.show()
