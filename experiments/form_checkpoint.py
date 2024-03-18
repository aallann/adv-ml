import os
import torch
import pandas as pd

from fcnn.model import FCNN
from fcnn.datasets import load_optim_datasets
from fcnn.utils import jacobian_approx, jacobian_approx_batchwise, param_init

device = "cuda" if torch.cuda.is_available() else "cpu"
lambda_: float = 1e-3
tol: float = 1e-8
lambda_: float = 1e-3
rho1: float = 0.25
rho2: float = 0.75
beta: float = 2.0
gamma: float = 3.0
max_iter: int = 120
n_iter: int = 0

train_loader, val_loader, data_processor = load_optim_datasets(
    data_path="data",
    contract="call",
    n_train=390000,
    train_batch_size=16,
    test_batch_size=2048,
    n_features=7,
    input_encoder=True,
    output_encoder=False,
)

data_processor.to(device)

# Initialize the model
model = FCNN(in_dim=7, out_dim=1, hidden_dim=64, n_layers=12, device=device)

# Load the saved model
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "models", "model.pt"
)
model.load_state_dict(torch.load(save_path))
model.to(device)

# Validation on implied volatility surface data
data = val_loader.dataset
synth: torch.Tensor = (
    torch.stack([sample["x"][:2] for sample in data]).type(torch.float32).to(device)
)
quotes: torch.Tensor = (
    torch.stack([sample["y"] for sample in data]).type(torch.float32).to(device)
)
del data

p_init = param_init()

n_samples: int = synth.shape[0]
n_params: int = p_init.shape[0]

x_init: torch.Tensor = torch.concatenate(
    [synth, p_init.repeat(n_samples, 1)], dim=1
).to(device)

if data_processor is not None:
    x_init = data_processor.input_encoder.transform(x_init)

# Feed the entire dataset to the model at once
iv_nn = model(x_init)
iv_nn: torch.Tensor = model(x_init)
I: torch.Tensor = torch.eye(n_params, device=device)
J = jacobian_approx_batchwise(x_init, model)[:, 2:]
residuals: torch.Tensor = quotes - iv_nn
delta: torch.Tensor = (
    torch.pinverse(J.t() @ J + lambda_ * I) @ J.t() @ residuals
).flatten()
"""
L = torch.cholesky(J.t() @ J + lambda_ * I)
y = torch.cholesky_solve(J.t() @ residuals, L)
delta = y.flatten()"""

while residuals.norm() > tol and n_iter < max_iter:
    p_k: torch.Tensor = p_init + delta
    x_k: torch.Tensor = torch.concatenate([synth, p_k.repeat(n_samples, 1)], dim=1).to(
        device
    )

    if data_processor is not None:
        x_k = data_processor.input_encoder.transform(x_k)

    # x_k.requires_grad_(True)
    iv_nn_k = model(x_k)
    r_k = quotes - iv_nn_k
    print(f"Residuals: {r_k.norm()}")
    J_k = jacobian_approx_batchwise(x_k, model)[:, 2:]
    rnorm_k = r_k.norm()
    c_mu = (rnorm_k + residuals.norm()) / (
        residuals.norm() - (residuals + J @ delta).norm()
    )

    if c_mu <= rho1:
        lambda_ *= beta

    else:
        lambda_ /= beta

    if c_mu >= rho2:
        lambda_ /= beta

    delta = (torch.pinverse(J_k.t() @ J_k + lambda_ * I) @ J_k.t() @ r_k).flatten()
    """L = torch.cholesky(J.t() @ J + lambda_ * I)
    y = torch.cholesky_solve(J.t() @ residuals, L)
    delta = y.flatten()"""

    n_iter += 1
    p_init = p_k

# concatenate a tensor with two entries 0, and p_k
p_k = torch.cat((torch.zeros(2), p_k))

print(J.shape)
print(residuals.shape)
print(delta.shape)
print(delta[:10])
print(p_k)
print(data_processor.input_encoder.inverse_transform(p_k.reshape(1, -1)))


"""x_subset = x_init[:]
jac_approx1 = jacobian_approx_batchwise(x_subset, model).squeeze()
jac_approx2 = jacobian_approx(x_subset, model).squeeze()


# compare element wise to see if they are the same
# print(torch.allclose(jac_approx1, jac_approx2, atol=1e-6))
#print(jac_approx1.shape)
#print(jac_approx2.shape)
#print(x_init[:10])
#print(predictions[:10])"""
