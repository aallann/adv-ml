import abc
import pandas as pd
import numpy as np
from datetime import datetime


class Transform(abc.ABC):
    """Transform base class"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def transform(self, data_frame: pd.DataFrame):
        pass

    @abc.abstractmethod
    def inverse_transform(self, data_frame: pd.DataFrame):
        pass


class Composite(Transform):
    """Composite transform class for multiprocess"""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, data_frame: pd.DataFrame):
        for transform in self.transforms:
            data_frame = transform.transform(data_frame)

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame):
        for transform in reversed(self.transforms):
            data_frame = transform.inverse_transform(data_frame)

        return data_frame


class DateFilter(Transform):
    """Date filter"""

    def __init__(self):
        super().__init__()
        self.field: str = "expiration_date"

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame[
            data_frame[self.field].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
        ]
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Obviate inverse data transform, is inexistent"""
        return data_frame


class LogScaler(Transform):
    """Logarithmic scaler"""

    def __init__(self):
        super().__init__()
        self.field: str = "moneyness"

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame[f"log_{self.field}"] = data_frame[self.field].apply(np.log)
        data_frame = data_frame.drop(columns=[self.field])
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame[self.field] = data_frame[f"log_{self.field}"].apply(np.exp)
        data_frame = data_frame.drop(columns=[f"log_{self.field}"])
        return data_frame.dropna()


class LiquidityFilter(Transform):
    """Liquidity filter"""

    def __init__(self):
        super().__init__()
        self.open_int: str = "openInterest"
        self.time_to_maturity: str = "time_to_maturity"
        self.log_moneyness: str = "log_moneyness"

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame[data_frame[self.open_int] != 0]
        data_frame = data_frame[data_frame[self.time_to_maturity] > 1 / 365]
        data_frame = data_frame[data_frame[self.time_to_maturity] < 0.2]
        data_frame = data_frame[data_frame[self.log_moneyness] > -0.1]
        data_frame = data_frame[data_frame[self.log_moneyness] < 0.28]
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Obviate inverse data transform, is inexistent"""
        return data_frame


class AppendTau(Transform):
    """Append Tau to data frame"""

    def __init__(self):
        super().__init__()
        self.date = datetime.today().strftime("%Y-%m-%d")
        self.field: str = "expiration_date"

    def get_time_to_maturity(self, data_frame: pd.DataFrame):
        """Get time to maturity for all options contracts in database"""
        opt_expiries: pd.DataFrame = pd.to_datetime(
            data_frame[self.field], format="%Y-%m-%d"
        )

        date = pd.to_datetime(self.date, format="%Y-%m-%d", errors="coerce")

        times_to_expiry: pd.DataFrame = (
            (opt_expiries.apply(lambda exp: (exp - date).days) / 365)
            .to_frame()
            .rename(columns={self.field: "time_to_maturity"})
        )

        return times_to_expiry

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        taus = self.get_time_to_maturity(data_frame)
        return pd.concat([data_frame, taus], axis=1)

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Obviate inverse data transform, is inexistent"""
        return data_frame


class AppendMid(Transform):
    """Append mid price to data frame"""

    def __init__(self):
        super().__init__()
        self.field: str = "mid"

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame[self.field] = (data_frame["bid"] + data_frame["ask"]) / 2
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Obviate inverse data transform, is inexistent"""
        return data_frame


class DropElse(Transform):
    """Drop all columns except for the ones specified"""

    def __init__(self):
        super().__init__()
        self.fields = [
            "strike",
            "openInterest",
            "bid",
            "ask",
            "mid",
            "log_moneyness",
            "time_to_maturity",
        ]

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame[self.fields]
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Obviate inverse data transform, is inexistent"""
        return data_frame


class ReComp(Transform):
    """Reweights proportional composition of database contracts based on open interest volume.
    Establishes contract densities in dataset to reflect market conditions.
    """

    def __init__(self):
        super().__init__()
        self.field = "openInterest"

    def volume_rescale_composition(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame.reset_index(drop=True)
        total_int = data_frame[self.field].sum()

        # Convert DataFrame columns to numpy arrays
        time_to_maturity = data_frame["time_to_maturity"].values
        log_moneyness = data_frame["log_moneyness"].values
        open_interest = data_frame["openInterest"].values

        # Create placeholder numpy arrays
        time_to_maturity_placeholder = np.zeros(total_int)
        log_moneyness_placeholder = np.zeros(total_int)

        counter = 0
        for i in range(len(data_frame)):
            num_int = 1
            values_time_to_maturity = np.full(num_int, time_to_maturity[i])
            values_log_moneyness = np.full(num_int, log_moneyness[i])

            time_to_maturity_placeholder[counter : counter + num_int] = (
                values_time_to_maturity
            )
            log_moneyness_placeholder[counter : counter + num_int] = (
                values_log_moneyness
            )

            counter += num_int

        # Convert numpy arrays back to DataFrame
        placeholder = pd.DataFrame(
            {
                "time_to_maturity": time_to_maturity_placeholder,
                "log_moneyness": log_moneyness_placeholder,
            }
        )

        return placeholder

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        # data_frame = self.volume_rescale_composition(data_frame)
        data_frame = data_frame.dropna()
        return data_frame[["time_to_maturity", "log_moneyness"]]

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Obviate inverse data transform, is inexistent"""
        return data_frame
