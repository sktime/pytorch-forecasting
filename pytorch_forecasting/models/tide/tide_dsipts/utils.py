import torch


def get_scope(
    handle_multivariate,
    handle_future_covariates,
    handle_categorical_variables,
    handle_quantile_loss,
):
    message = (
        f'Can {"NOT" if not handle_multivariate else ""}  '
        f'handle multivariate output \n'
        f'Can {"NOT" if not handle_future_covariates else ""}  '
        f'handle future covariates\n'
        f'Can {"NOT" if not handle_categorical_variables else ""}  '
        f'handle categorical covariates\n'
        f'Can {"NOT" if not handle_quantile_loss else ""}  '
        f'handle Quantile loss function'
    )

    return message


def beauty_string(message: str, type: str, verbose: bool):
    import logging

    size = 150
    if verbose is True:
        if type == "block":
            characters = len(message)
            border = max((100 - characters) // 2 - 5, 0)
            logging.info("\n")
            logging.info(f"{'#' * size}")
            logging.info(f"{'#' * border}{' ' * (size - border * 2)}{'#' * border}")
            logging.info(f"{message:^{size}}")
            logging.info(f"{'#' * border}{' ' * (size - border * 2)}{'#' * border}")
            logging.info(f"{'#' * size}")
        elif type == "section":
            logging.info("\n")
            logging.info(f"{'#' * size}")
            logging.info(f"{message:^{size}}")
            logging.info(f"{'#' * size}")
        elif type == "info":
            logging.info(f"{message:^{size}}")
        else:
            logging.info(message)


class SinkhornDistance:
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction="none"):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def compute(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).to(x.device)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .squeeze()
            .to(x.device)
        )
        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .squeeze()
            .to(x.device)
        )

        u = torch.zeros_like(mu).to(x.device)
        v = torch.zeros_like(nu).to(x.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost  # , pi, C
