import torch
import pandas as pd


def tensorize(
    data_frame: pd.DataFrame, save_name: str = "Calls", split: bool = True
) -> None:
    """
    Tensorizes pandas DataFrame (pd.DataFrame -> torch.Tensor(dtype=float32))
    and stores as a .pt file. If split is True, the dataset is split into
    training and testing datasets and stored as separate .pt files.

    Args
    ----
        :param dataset: pandas DataFrame
        :param split: bool
    """
    data_frame.dropna(inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame = data_frame[data_frame["IV"] != 0]

    tensor: torch.Tensor = torch.tensor(data_frame.values, dtype=torch.float32)

    x: torch.Tensor = tensor[:, :-1]
    y: torch.Tensor = tensor[:, -1]

    tensor_dataset = torch.utils.data.TensorDataset(x, y)

    if split:
        total_size = len(tensor_dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size

        train_set, test_set = torch.utils.data.random_split(
            tensor_dataset, [train_size, test_size]
        )

        train_dict = {"x": train_set[:][0], "y": train_set[:][1]}
        test_dict = {"x": test_set[:][0], "y": test_set[:][1]}

        torch.save(train_dict, f"data/tensors/Heston{save_name}Train.pt")
        torch.save(test_dict, f"data/tensors/Heston{save_name}Test.pt")

    else:
        data_set = {"x": tensor_dataset[:][0], "y": tensor_dataset[:][1]}
        torch.save(data_set, f"data/tensors/HestonUnsplit{save_name}.pt")
