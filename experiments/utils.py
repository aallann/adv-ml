import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_implied_vol_surface(data_dict: dict):

    x = data_dict["x"].to("cpu").numpy()
    y = data_dict["y"].to("cpu").numpy()

    log_moneyness = x[:, 0]
    time_to_maturity = x[:, 1]
    iv = y

    X, Y = np.meshgrid(log_moneyness, time_to_maturity)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        log_moneyness,
        time_to_maturity,
        iv,
        cmap="inferno",
        edgecolor="none",
    )

    ax.set_xlabel("Log Moneyness")
    ax.set_ylabel("Time to Maturity")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Implied Volatility Surface")

    fig.colorbar(ax.plot_trisurf(X, Y, iv, cmap="inferno"), ax=ax)

    plt.show()
