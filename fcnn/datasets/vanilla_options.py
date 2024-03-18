from pathlib import Path
import torch

from .transforms import UnitGaussianNormaliser
from .processing import DataProcessor
from ..utils import TensorDictDataset


def load_vanilla_derivative_contracts(
    data_path: str,
    contract: str,
    n_train: int,
    n_test: int,
    train_batch_size: int,
    test_batch_size: int,
    n_features: int,
    input_encoder: bool = True,
    output_encoder: bool = False,
):
    """Load vanilla derivatives contracts from synthetic dataset

    Args
    ----
        :param database: name of the database
        :param contract: call/put
        :param ntrain: number of training samples
        :param ntest: number of testing samples
        :param train_batch_size: size of training batches
        :param test_batch_size: size of testing batches
        :param nfeatures: number of input features
        :param input_encoder: bool, default is True
        :param output_encoder: bool, default is False

    Returns
    -------
        train_loader, test_loader, data_processor
    """
    _DIR = Path(__file__).parent

    data = torch.load(
        _DIR.joinpath(data_path, f"Heston{contract.lower().title()}sTrain.pt")
    )

    x_train = data["x"][:n_train, :].type(torch.float32)
    y_train = data["y"][:n_train].unsqueeze(-1).type(torch.float32)
    del data

    data = torch.load(
        _DIR.joinpath(data_path, f"Heston{contract.lower().title()}sTest.pt")
    )

    x_test = data["x"][:n_test, :].type(torch.float32)
    y_test = data["y"][:n_test].unsqueeze(-1).type(torch.float32)
    del data

    data = torch.load(
        _DIR.joinpath(data_path, f"HestonUnsplit{contract.lower().title()}s.pt")
    )

    x_val = data["x"][:, :].type(torch.float32)
    y_val = data["y"][:].unsqueeze(-1).type(torch.float32)
    del data

    if input_encoder:
        input_encoder = UnitGaussianNormaliser()
        input_encoder.fit(x_train)
        # x_train = input_encoder.transform(x_train)
        # x_test = input_encoder.transform(x_test)
        # x_val = input_encoder.transform(x_val)

    if output_encoder:
        output_encoder = UnitGaussianNormaliser()
        output_encoder.fit(y_train)
        # y_train = output_encoder.transform(y_train)
        # y_test = output_encoder.transform(y_test)
        # y_val = output_encoder.transform(y_val)

    train_dataset = TensorDictDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_dataset = TensorDictDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = TensorDictDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    data_processor = DataProcessor(
        input_encoder=input_encoder,
        output_encoder=output_encoder,
        n_features=n_features,
    )

    return train_loader, test_loader, val_loader, data_processor


def load_optim_datasets(
    data_path: str,
    contract: str,
    n_train: int,
    train_batch_size: int,
    test_batch_size: int,
    n_features: int,
    input_encoder: bool = True,
    output_encoder: bool = False,
):
    """Load vanilla options datasets for optimisation algorithm.

    Args
    ----
        :param database: name of the database
        :param contract: call/put
        :param ntrain: number of training samples
        :param ntest: number of testing samples
        :param train_batch_size: size of training batches
        :param test_batch_size: size of testing batches
        :param nfeatures: number of input features
        :param input_encoder: bool, default is True
        :param output_encoder: bool, default is False

    Returns
    -------
        train_loader, test_loader, data_processor
    """

    _DIR = Path(__file__).parent

    data = torch.load(
        _DIR.joinpath(data_path, f"Heston{contract.lower().title()}sTrain.pt")
    )

    x_train = data["x"][:n_train, :].type(torch.float32)
    y_train = data["y"][:n_train].unsqueeze(-1).type(torch.float32)
    del data

    data = torch.load(_DIR.joinpath(data_path, "HestonUnsplitOptim.pt"))

    x_val = data["x"][:100, :].type(torch.float32)
    y_val = data["y"][:100].unsqueeze(-1).type(torch.float32)
    print(x_val.shape)
    print(y_val.shape)
    del data

    if input_encoder:
        input_encoder = UnitGaussianNormaliser()
        input_encoder.fit(x_train)

    if output_encoder:
        output_encoder = UnitGaussianNormaliser()
        output_encoder.fit(y_train)

    train_dataset = TensorDictDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = TensorDictDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    data_processor = DataProcessor(
        input_encoder=input_encoder,
        output_encoder=output_encoder,
        n_features=n_features,
    )

    return train_loader, val_loader, data_processor
