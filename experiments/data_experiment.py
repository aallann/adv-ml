import os
import torch

from fcnn.datasets import load_optim_datasets


device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, data_processor = load_optim_datasets(
    data_path="data",
    contract="call",
    n_train=300000,
    train_batch_size=16,
    test_batch_size=2048,
    n_features=7,
    input_encoder=True,
    output_encoder=False,
)


# Quicktests


test_samples = val_loader.dataset

for idx in range(3):
    data = test_samples[idx]
    data = data_processor.preprocess(data)

    print(data["x"].shape, data["y"].shape)

    x = data["x"].to(device)
    y = data["y"].to(device)
    test = data["x"][0]
    print(test)
    print(test.shape)
    print(x, y)
    print(x.shape, y.shape)

test_sample = test_samples[0]  # Get the first sample from the dataset
print(test_sample["x"][1].shape)  # Now you can use string indexing
