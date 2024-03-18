from database import OptionsDatabase
from data_processor import DataProcessor


def load_vanilla_options_dataframes(
    database: str,
    params: dict = None,
    input_encoder=None,
    output_encoder=False,
    verbose: bool = False,
):
    if params is None:
        params = {
            "ticker": r"^SPX",
            "date": "2024-02-01",
            "period": "1d",
        }

        if verbose:
            print(f"Using default parameters: {params}")

    else:
        assert params.keys() == {"ticker", "date", "period"}, "Invalid parameters"

    tables = [
        f"{params['ticker'].replace('^', '')}_calls",
        f"{params['ticker'].replace('^', '')}_puts",
        "SPOT",
    ]
    data_processor = DataProcessor()

    database = OptionsDatabase(database=database)

    calls = database.read(tables[0], "1=1")
    puts = database.read(tables[1], "1=1")
    state_dict = database.read(tables[2], "1=1")

    spot = state_dict["spot_price"].values[0]

    calls["moneyness"] = calls["strike"] / spot
    puts["moneyness"] = puts["strike"] / spot

    if input_encoder:
        data_processor = DataProcessor(
            input_encoder=input_encoder, output_encoder=output_encoder
        )
        calls = data_processor.preprocess(calls)
        puts = data_processor.preprocess(puts)

    if output_encoder:
        data_processor = DataProcessor(
            input_encoder=input_encoder, output_encoder=output_encoder
        )
        calls = data_processor.postprocess(calls)
        puts = data_processor.postprocess(puts)

    return calls, puts, data_processor
