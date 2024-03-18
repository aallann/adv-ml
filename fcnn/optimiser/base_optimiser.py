import abc
import torch
from typing import Dict, Callable

from ..utils import jacobian_approx
from ..datasets.processing import DataProcessor


class Optimizer(torch.optim.Optimizer):
    """Least Squares curve fitting Heston stochastic 
    volatility model parameters to synthetic market data
    
    Employs Levenverg-Marquardt optimisation algorithm:
    minimising the sum of weighted squared residuals.

    Args
    ----
        :param p_init: initial input values
        :param model: trained model IV mapping (acts as the model function)
        :param jac: Jacobian approximation of model function of inputs wrt output
        :param data: model data (time to maturity, true IV, log moneyness)
        :param mtol: stopping criterion tolerance for relative model convergence
        :param gtol: stopping criterion tolerance for relative gradient convergence
        :param ptol: stopping criterion tolerance for relative parameter convergence
        :param lr: (learning rate) since LM, initial damping factor
        :param rho1: first gain threshold for damping adjust
        :param rho2: second gain threshold for damping adjust
        :param beta: scale factor for damping adjust (mult)
        :param gamma: scale factor for damping adjust (div)
        :param max_iter: maximum number of iterations
    """

    def __init__(
        self,
        p_init: torch.Tensor,
        model: torch.nn.Module,
        jac: jacobian_approx,
        mtol: float = 1e-8,
        gtol: float = 1e-8,
        ptol: float = 1e-8,
        lr: float = 1e-3,
        rho1: float = 0.25,
        rho2: float = 0.75,
        beta: float = 2.0,
        gamma: float = 3.0,
        max_iter: int = 100,
        device: str = "cpu",
        data_processor: DataProcessor = None,
    ):

        self.p_init = p_init
        self.model = model
        self.jac = jac
        self.tol = tol
        self._lambda = _lambda
        self.rho1 = rho1
        self.rho2 = rho2
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter

    @abc.abstractmethod
    def step(self):
        """Optimiser step"""
        return NotImplementedError

    @abc.abstractmethod
    def update(self):
        """Optimiser update"""
        return NotImplementedError


class LevenverMarquardtOptimiser(Optimizer):
    """Levenver-Marquardt optimiser for non-linear least squares problems

    Args
    ----
        :param p: initial input values
        :param model: trained model IV mapping (acts as the model function)
        :param jac: Jacobian approximation of model function of inputs wrt output
        :param data: model data (time to maturity, true IV, log moneyness)
        :params init_params: initial stochastic volatility model parameters
        :param wvec: weights vector for loss function
        :param tol: tolerance for convergence
        :param _lambda: initial damping factor
        :param rho1: first gain threshold for damping adjust
        :param rho2: second gain threshold for damping adjust
        :param beta: scale factor for damping adjust (mult)
        :param gamma: scale factor for damping adjust (div)
        :param max_iter: maximum number of iterations
    """

    def __init__(
        self,
        p_init: torch.Tensor,
        model: torch.nn.Module,
        jac: jacobian_approx,
        data_loader=None,
        data_processor=None,
        init_params=None,
        wvec: torch.Tensor = None,
        tol: float = 1e-8,
        _lambda: float = 1e-3,
        rho1: float = 0.25,
        rho2: float = 0.75,
        beta: float = 2.0,
        gamma: float = 3.0,
        max_iter: int = 100,
        device: str = "cpu",
    ):

        super().__init__(
            p=p_init,
            model=model,
            jac=jac,
            wvec=wvec,
            tol=tol,
            _lambda=_lambda,
            rho1=rho1,
            rho2=rho2,
            beta=beta,
            gamma=gamma,
            max_iter=max_iter,
        )

        self.data_loader = data_loader
        self.data_processor = data_processor
        self.moneyness = data["x"][:, 0]
        self.tau = data["x"][:, 1]
        self.iv = data["y"]

        self.init_params = init_params

        self.device = device
        self.model.eval()
        self.model.to(self.device)

        self.iter = 0
        self.history = {
            "deltas": {k: [] for k in self.init_params.keys()},
            "residuals": [],
            "lambdas": [],
        }

    def step(self, p: torch.Tensor):
        """Optimiser step

        Args
        ----
            :param p: input values
        """
    
    def build_dataset(self, params: torch.Tensor):
        """Build input tensors for model"""

        for idx, sample in enumerate(self.data_loader):
            

