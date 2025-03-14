import matplotlib.pyplot as plt
import numpy as np
import torch
import os 






def save_backtest_plot(DM, portfolio_history, portfolio_distribution, csv_dir):
    test_indices = DM._test_ind

    x_portfolio = range(len(portfolio_history))
    xlabel = "Test Steps"


    # Extract the close prices from the global panel.
    # The panel is organized as [feature, coin, time], so get the "close" prices.
    close_prices = DM.global_matrix.loc['close']
    # Transpose so that the index is time and the columns are crypto symbols.
    close_prices = close_prices.T.sort_index()
    
    # Use the test indices to extract the corresponding time stamps.
    test_times = close_prices.index[test_indices[0]: test_indices[-1]+1]
    close_prices_test = close_prices.loc[test_times]

    normalized_close = close_prices_test / close_prices_test.iloc[0]
    #####
 
    x_crypto = range(len(test_times)) if len(x_portfolio) != len(test_times) else test_times

    if isinstance(portfolio_distribution, torch.Tensor):
        portfolio_distribution = portfolio_distribution.cpu().numpy()
    if len(portfolio_distribution.shape) == 1:
        portfolio_distribution = portfolio_distribution[:, None]

    labels = list(normalized_close.columns) 

    fig, ax = plt.subplots(4, 1, figsize=(12, 20))

    ax[0].plot(x_portfolio, portfolio_history, marker='o', label="Portfolio Cumulative Return", color='blue')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("Cumulative Return")
    ax[0].set_title("Backtest Portfolio Performance Evolution")
    ax[0].legend()
    ax[0].grid(True)

    for coin in normalized_close.columns:
        ax[1].plot(x_crypto, normalized_close[coin], label=coin)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("Cumulative Return")
    ax[1].set_title("Backtest Individual Crypto Performance Evolution")
    ax[1].legend(loc='best', fontsize='small')
    ax[1].grid(True)

    x_distribution = np.arange(portfolio_distribution.shape[0])
    ax[2].stackplot(x_distribution, portfolio_distribution.T, labels=labels)
    ax[2].set_xlabel("Time Steps")
    ax[2].set_ylabel("Portfolio Weights")
    ax[2].set_title("Portfolio Distribution Evolution")
    ax[2].legend(loc='upper left', fontsize='small')
    ax[2].grid(True)

    zoom_start = max(0, portfolio_distribution.shape[0] - 100)
    ax[3].stackplot(x_distribution[zoom_start:], portfolio_distribution[zoom_start:].T, labels=labels)
    ax[3].set_xlabel("Time Steps")
    ax[3].set_ylabel("Portfolio Weights")
    ax[3].set_title("Zoomed Portfolio Distribution (Last 100 Time Steps)")
    ax[3].legend(loc='upper left', fontsize='small')
    ax[3].grid(True)

    plt.tight_layout()
    folder = os.path.dirname(csv_dir)
    fig_path = os.path.join(folder, "backtest_test_history.png")
    plt.savefig(fig_path)
    plt.close()
    print("Backtest performance plot saved to:", fig_path)
