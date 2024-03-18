from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException

import logging
import datetime
import numpy as np
import QuantLib as qlib

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def heston_pricer(
    kappa: float,
    theta: float,
    vov: float,
    rho: float,
    sigma: float,
    r: float,
    q: float,
    tau: float,
    S: float,
    K: float,
) -> tuple:
    """Heston pricer

    Args
    ----
    :param kappa: reversion rate toward mean volatility
    :param theta: mean/long-term equilibrium volatility
    :param vov: variance of volatility
    :param rho: correlation between asset and volatility
    :param sigma: initial/spot volatility
    :param r: risk-free rate
    :param q: dividend yield
    :param tau: time to maturity
    :param S: spot price
    :param K: strike price

    Output
    ------
    :return: option price
    :return: implied voltility
    """

    date = datetime.date.today()
    date = qlib.Date(date.day, date.month, date.year)
    count = qlib.Actual365Fixed()
    qlib.Settings.instance().evaluationDate = date

    option = qlib.Option.Call
    payoff = qlib.PlainVanillaPayoff(option, K)
    maturity = date + int(tau * 365)
    exercise = qlib.EuropeanExercise(maturity)
    european_option = qlib.VanillaOption(payoff, exercise)

    quote_handle = qlib.QuoteHandle(qlib.SimpleQuote(S))
    flat_term_struct = qlib.YieldTermStructureHandle(qlib.FlatForward(date, r, count))
    dividend_yield = qlib.YieldTermStructureHandle(qlib.FlatForward(date, q, count))
    heston = qlib.HestonProcess(
        flat_term_struct, dividend_yield, quote_handle, sigma, kappa, theta, vov, rho
    )

    engine = qlib.AnalyticHestonEngine(qlib.HestonModel(heston), 1e-15, int(1e6))
    european_option.setPricingEngine(engine)

    try:
        price = european_option.NPV()
        if price <= 0 or price + K < S:
            iv = np.nan
            logging.debug(
                "NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(
                    price, S - K, tau, K
                )
            )
        else:
            logging.debug("Success: Price {} > intrinsic {}".format(price, S - K))
            iv = implied_volatility(price, S, K, tau, r, "c")
    except RuntimeError:
        logging.info(
            "RuntimeError: Intrinsic {}. Time {}. Strike {}.".format(S - K, tau, K)
        )
        price = np.nan
        iv = np.nan

    return price, iv
