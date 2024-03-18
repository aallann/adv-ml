import torch


class TensorDictDataset(torch.utils.data.Dataset):
    """TensorDictDataset

    Args
    ----
        :param data_dict: dictionary of tensors
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        assert x.size(0) == y.size(0), "x and y must same first dimension"
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return {"x": x, "y": y}

    def __len__(self):
        return self.x.size(0)


def jacobian_approx(p: torch.tensor, model: torch.nn.Module) -> torch.Tensor:
    """Compute model function Jacobian approximation for some input

    [strict = True, vectorize = False] is employed in this
    case due to the fact that the model function is a mapping
    from higher dimensional spaces to a scalar (R^7 -> R^1).
    Otherwise, vectorise=True would be faster...

    Args
    ----
        :param p: input model eval values
        :param model: model (acting as model function)

    Returns
    -------
        jac: torch.tensor, Jacobian approximation

    jac = torch.autograd.functional.jacobian(
        func=lambda p_: model(p_),
        inputs=p,
        create_graph=True,
        strict=True,
        vectorize=False,
    )"""
    jac_list = []
    for p_ in p:
        jac = torch.autograd.functional.jacobian(
            func=lambda p__: model(p__.unsqueeze(0)),
            inputs=p_,
            create_graph=True,
            strict=True,
            vectorize=False,
        )
        jac_list.append(jac.squeeze())
    return torch.stack(jac_list)


def jacobian_approx_batchwise(
    inputs: torch.Tensor, model: torch.nn.Module, batch_size: int = 32
) -> torch.Tensor:
    """Compute model function Jacobian approximation for some input.

    [strict = True, vectorize = False] is employed in this
    case due to the fact that the model function is a mapping
    from higher dimensional spaces to a scalar (R^7 -> R^1).
    Otherwise, vectorise=True would be faster...

    Args
    ----
        :param p: input model eval values
        :param model: model (acting as model function)

    Returns
    -------
        jac: torch.tensor, Jacobian approximation
    """
    n = inputs.size(0)
    jac_list = []
    for i in range(0, n, batch_size):
        batch_inputs = inputs[i : i + batch_size]
        for input in batch_inputs:
            jac = torch.autograd.functional.jacobian(
                func=lambda input_: model(input_.unsqueeze(0)),
                inputs=input,
                create_graph=True,
                strict=True,
                vectorize=False,
            )
            jac_list.append(jac.squeeze())
    return torch.stack(jac_list)


def param_init():
    """Heston model parameter initialiser

    Returns
    -------
        params: torch.Tensor dictionary with pseudo-random Heston model parameters"""

    kappa: torch.Tensor = 3 * torch.rand(1)
    theta: torch.Tensor = 0.5 * torch.rand(1)
    vov: torch.Tensor = 1 * torch.rand(1)
    rho: torch.Tensor = -1 * torch.rand(1)
    sigma: torch.Tensor = 0.5 * torch.rand(1)

    params = {
        "kappa": kappa,
        "theta": theta,
        "vov": vov,
        "rho": rho,
        "sigma": sigma,
    }

    return torch.Tensor([params[k] for k in params.keys()])
