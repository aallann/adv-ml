from data_processor import DataProcessor
from kde import KernelDensityEstimator
from heston_solver import heston_pricer

import numpy as np
import pandas as pd


def synthesize_vanilla_options_data(
    n_samples: int,
    data_frame: pd.DataFrame,
    input_encoder=None,
    output_encoder=None,
    kde=KernelDensityEstimator,
):
    """Synthesise vanilla options parameter data
    according to Heston Stochastic Volatility Model

    Args
    ----
        :param n_samples: number of samples to syntehtise
        :param data_frame: input data frame
        :param input_encoder: input encoder
        :param output_encoder: output encoder
        :param kde: kernel density estimator
        :param bandwidth: bandwidth for the kernel density estimator
        :param device: device to use for the kernel density estimator (if pytorch)
        :param verbose: verbosity
    """

    if input_encoder:
        data_processor = DataProcessor(input_encoder=input_encoder, output_encoder=None)
        data_frame = data_processor.preprocess(data_frame)

    if output_encoder:
        data_processor = DataProcessor(
            input_encoder=None, output_encoder=output_encoder
        )
        data_frame = data_processor.postprocess(data_frame)

    data_frame = data_frame[["moneyness", "time_to_maturity"]]
    print(f"data_frame columns: {data_frame.columns}")
    print(data_frame.head())

    # data_frame = data_frame.dropna()
    kde = kde(data_frame)
    kde.fit()
    samples = kde.sample(n_samples=n_samples)
    del data_frame

    # Debugging
    print(f"samples columns: {samples.columns}")
    print(samples.head())

    data_frame = pd.DataFrame(
        index=np.arange(n_samples),
        columns=["kappa", "theta", "vov", "rho", "sigma", "IV"],
    )

    data_frame = pd.concat([samples, data_frame], axis=1)

    params: dict = {
        "kappa": [0.0, 4.0],
        "theta": [0.0, 0.5],
        "vov": [0.0, 1.0],
        "rho": [-1.0, 0.0],
        "sigma": [0.0, 0.5],
        "S": 1,
        "r": 0.0,
        "q": 0.0,
    }

    for column in ["kappa", "theta", "vov", "rho", "sigma"]:
        data_frame[column] = np.random.uniform(
            low=params[column][0], high=params[column][1], size=len(samples)
        )

    data_frame["IV"] = data_frame.apply(
        lambda row: heston_pricer(
            row["kappa"],
            row["theta"],
            row["vov"],
            row["rho"],
            row["sigma"],
            params["r"],
            params["q"],
            row["time_to_maturity"],
            params["S"],
            row["moneyness"],
        )[1],
        axis=1,
    )

    return data_frame.dropna()


def synthesize_vanilla_options_smile(
    n_samples: int = 1000,
    params: dict = None,
):
    """Synthesises smile data for vanilla options."""

    if not params:
        params = {
            "kappa": 1.3253,
            "theta": 0.0354,
            "vov": 0.3877,
            "rho": -0.7165,
            "sigma": 0.0174,
            "S": 1,
            "r": 0.0,
            "q": 0.0,
        }

    log_moneyness: np.array = np.linspace(-0.1, 0.28, n_samples)
    strikes: np.array = params["S"] * np.exp(log_moneyness)

    cols: list = ["strikes", "log_moneyness", "IV"]

    data_frame: pd.DataFrame = pd.DataFrame(
        np.zeros((n_samples, len(cols))), columns=cols
    )

    data_frame["strikes"] = strikes
    data_frame["log_moneyness"] = log_moneyness

    for i in range(n_samples):
        data_frame["IV"][i] = heston_pricer(
            params["kappa"],
            params["theta"],
            params["vov"],
            params["rho"],
            params["sigma"],
            params["r"],
            params["q"],
            1,
            params["S"],
            strikes[i],
        )[1]

    return data_frame.dropna()


def synthesize_vanilla_options_surface(
    n_samples: int = 200,
    params: dict = None,
):
    """Synthesises surface dataset for vanilla options.

    Mesh time to maturity and moneyness, params are static"""

    if not params:
        params = {
            "kappa": 1.3253,
            "theta": 0.0174,
            "vov": 0.3877,
            "rho": -0.7165,
            "sigma": 0.0354,
            "S": 1,
            "r": 0.0,
            "q": 0.0,
        }

    time_to_maturity: np.array = np.linspace(0.1, 0.18, n_samples)
    log_moneyness: np.array = np.linspace(-0.1, 0.1, n_samples)

    cols: list = ["strikes", "moneyness", "time_to_maturity", "IV"]

    data_frame: pd.DataFrame = pd.DataFrame(
        np.zeros((n_samples**2, len(cols))), columns=cols
    )

    X, Y = np.meshgrid(time_to_maturity, log_moneyness)

    data_frame["moneyness"] = np.exp(Y.flatten())
    data_frame["time_to_maturity"] = X.flatten()
    data_frame["strikes"] = params["S"] / data_frame["moneyness"].values
    strikes = data_frame["strikes"].values
    time_to_maturity = data_frame["time_to_maturity"].values

    for i in range(len(data_frame.index)):
        data_frame["IV"][i] = heston_pricer(
            params["kappa"],
            params["theta"],
            params["vov"],
            params["rho"],
            params["sigma"],
            params["r"],
            params["q"],
            time_to_maturity[i],
            params["S"],
            strikes[i],
        )[1]

    data_frame["kappa"] = np.ones_like(data_frame["IV"]) * params["kappa"]
    data_frame["theta"] = np.ones_like(data_frame["IV"]) * params["theta"]
    data_frame["vov"] = np.ones_like(data_frame["IV"]) * params["vov"]
    data_frame["rho"] = np.ones_like(data_frame["IV"]) * params["rho"]
    data_frame["sigma"] = np.ones_like(data_frame["IV"]) * params["sigma"]

    # rebuild dataset in the order moneyness, time_to_maturity, kappa, theta, vov, rho, sigma, IV
    data_frame = data_frame[
        ["moneyness", "time_to_maturity", "kappa", "theta", "vov", "rho", "sigma", "IV"]
    ]

    return data_frame.dropna()


def synthesize_vanilla_options_optim_data(
    n_samples: int = 500,
    params: dict = None,
):
    """Syntheizes Levenverg-Marquardt optimisation target dataset.

    It differs from the surface dataset only in that it employs
    time to maturity, moneyness gerated from the true distribution
    to reflect intraday market conditions; to this end, we reuse
    those generated in prior sections to avoid redundant computation.

    Args
    ----
        :param n_samples: number of samples to synthesise
        :param params: Heston model parameters
    """

    if not params:
        params = {
            "kappa": 1.3253,
            "theta": 0.0354,
            "vov": 0.3877,
            "rho": -0.7165,
            "sigma": 0.0174,
            "S": 1,
            "r": 0.0,
            "q": 0.0,
        }

    cols: list = ["strikes", "moneyness", "time_to_maturity", "IV"]

    data_frame: pd.DataFrame = pd.DataFrame(
        np.zeros((n_samples, len(cols))), columns=cols
    )

    placeholder: pd.DataFrame = pd.read_csv("data/csvs/heston_data.csv")
    data_frame.loc[:, "moneyness"] = placeholder["moneyness"].values[:n_samples]
    data_frame.loc[:, "time_to_maturity"] = placeholder["time_to_maturity"].values[
        :n_samples
    ]

    data_frame["strikes"] = params["S"] / data_frame["moneyness"].values
    strikes = data_frame["strikes"].values
    time_to_maturity = data_frame["time_to_maturity"].values

    data_frame["IV"] = np.zeros(len(data_frame.index))
    for i in range(len(data_frame.index)):
        data_frame.at[i, "IV"] = heston_pricer(
            params["kappa"],
            params["theta"],
            params["vov"],
            params["rho"],
            params["sigma"],
            params["r"],
            params["q"],
            time_to_maturity[i],
            params["S"],
            strikes[i],
        )[1]

    data_frame["kappa"] = np.ones_like(data_frame["IV"]) * params["kappa"]
    data_frame["theta"] = np.ones_like(data_frame["IV"]) * params["theta"]
    data_frame["vov"] = np.ones_like(data_frame["IV"]) * params["vov"]
    data_frame["rho"] = np.ones_like(data_frame["IV"]) * params["rho"]
    data_frame["sigma"] = np.ones_like(data_frame["IV"]) * params["sigma"]

    data_frame = data_frame[
        ["moneyness", "time_to_maturity", "kappa", "theta", "vov", "rho", "sigma", "IV"]
    ]

    return data_frame.dropna()
