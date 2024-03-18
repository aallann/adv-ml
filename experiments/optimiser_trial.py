import os
import torch

from fcnn.model import FCNN
from fcnn.datasets import load_optim_datasets
from fcnn.optimiser import parallel_levenverg_marquardt_batchwise
from fcnn.utils import jacobian_approx_batchwise, param_init

device = "cuda" if torch.cuda.is_available() else "cpu"

#############################################################################
#                               Load data

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


#############################################################################
#                                 Load model

model = FCNN(in_dim=7, out_dim=1, hidden_dim=64, n_layers=3, device=device)

save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "models", "model.pt"
)
model.load_state_dict(torch.load(save_path))


#############################################################################
#                          Optimisation routine

p_init = param_init()
jac = jacobian_approx_batchwise


synth, quotes, p_k, hist_dict = parallel_levenverg_marquardt_batchwise(
    p_init=p_init,
    model=model,
    jac=jac,
    data_loader=val_loader,
    data_processor=data_processor,
    device=device,
)
p_k = torch.cat([torch.zeros(2, device=device), p_k])
# p_k = data_processor.input_encoder.inverse_transform(p_k)

x_k: torch.Tensor = torch.concatenate(
    [synth, p_k[2:].repeat(synth.shape[0], 1)], dim=1
).to(device)

x_k = data_processor.input_encoder.transform(x_k)

iv_nn: torch.Tensor = model(x_k)

# compare quotes and iv_nn difference
print(quotes - iv_nn)

print(p_k)
