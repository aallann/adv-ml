import torch

from ..utils import jacobian_approx


def parallel_levenverg_marquardt_batchwise(
    p_init: torch.Tensor,
    model: torch.nn.Module,
    jac: jacobian_approx,
    data_loader=None,
    data_processor=None,
    tol: float = 1e-8,
    damp_factor: float = 0.1,
    rho1: float = 0.25,
    rho2: float = 0.75,
    beta: float = 2.0,
    gamma: float = 3.0,
    max_iter: int = 110,
    device: str = "cpu",
):
    # load, unify data
    data = data_loader.dataset
    synth: torch.Tensor = (
        torch.stack([sample["x"][:2] for sample in data]).type(torch.float32).to(device)
    )
    quotes: torch.Tensor = (
        torch.stack([sample["y"] for sample in data]).type(torch.float32).to(device)
    )
    del data

    # initialise utility variables
    n_iter: int = 0
    damp: float = damp_factor
    n_samples: int = synth.shape[0]
    heston_params: list = ["kappa", "theta", "vov", "rho", "sigma"]

    # model shift to gpu, inference
    model.to(device).eval()

    # build input dataset
    x_init: torch.Tensor = torch.concatenate(
        [synth, p_init.repeat(n_samples, 1)], dim=1
    ).to(device)

    x_init = data_processor.input_encoder.transform(x_init)

    # initialise optimisation
    p: torch.Tensor = p_init
    I: torch.Tensor = torch.eye(p_init.shape[0], device=device)
    J: torch.Tensor = jac(x_init, model)[:, 2:]
    # J = data_processor.input_encoder.inverse_transform(J)
    ivnn: torch.Tensor = model(x_init)
    res: torch.Tensor = ivnn - quotes
    delta: torch.Tensor = (
        torch.pinverse(J.t() @ J + damp * I) @ (J.t() @ res)
    ).flatten()

    # initialise history dictionary
    hist_dict: dict = {
        "lambdas": [],
        "residuals": [],
        "relative improvement": [],
        "deltas": {i: [] for i in heston_params},
    }

    # update dict
    hist_dict["residuals"].append(res.norm().item())
    hist_dict["lambdas"].append(damp)
    for i, param in enumerate(heston_params):
        hist_dict["deltas"][param].append(delta[i].item())

    # loop
    while res.norm() > tol and n_iter < max_iter:
        p_k: torch.Tensor = p - delta
        x_k: torch.Tensor = torch.concatenate(
            [synth, p_k.repeat(n_samples, 1)], dim=1
        ).to(device)

        x_k = data_processor.input_encoder.transform(x_k)

        ivnn_k: torch.Tensor = model(x_k)
        res_k: torch.Tensor = ivnn_k - quotes
        J_k: torch.Tensor = jac(x_k, model)[:, 2:]
        # J_k = data_processor.input_encoder.inverse_transform(J_k)[:, 2:]
        r_norm: torch.Tensor = res.norm()
        r_norm_k: torch.Tensor = res_k.norm()
        rel_imp: torch.Tensor = (r_norm - r_norm_k) / (
            r_norm - (res - J @ delta).norm()
        )

        if rel_imp <= rho1:
            damp *= beta

        if rel_imp >= rho2:
            damp /= beta
            res = res_k
            J = J_k

        delta = (torch.pinverse(J_k.t() @ J_k + damp * I) @ J_k.t() @ res).flatten()

        hist_dict["residuals"].append(r_norm_k.item())
        hist_dict["lambdas"].append(damp)
        hist_dict["relative improvement"].append(rel_imp.item())
        for i, param in enumerate(heston_params):
            hist_dict["deltas"][param].append(delta[i].item())

        n_iter += 1

    if n_iter == max_iter:
        print("Max iterations reached")

    # dimensional lift to be processed by data_processor
    print(res.norm())

    return synth, quotes, p_k, hist_dict
