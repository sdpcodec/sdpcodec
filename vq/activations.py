# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

import math
import warnings
from functools import lru_cache

import torch
from torch import nn, sin, pow
import torch.nn.functional as F
from torch.nn import Parameter

try:
    from .snake_lite_triton import (
        SnakeLiteTritonFunction,
        is_triton_snake_lite_available,
    )
except Exception:  # pragma: no cover - optional dependency
    SnakeLiteTritonFunction = None

    def is_triton_snake_lite_available() -> bool:
        return False


def _reduce_broadcast_gradient(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Sum broadcasted gradients back to the input tensor's original shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(dim=0)

    for dim, size in enumerate(target_shape):
        if size == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim=dim, keepdim=True)

    return grad.reshape(target_shape)


def _wrap_snake_lite_argument(z: torch.Tensor) -> torch.Tensor:
    """Wrap z to the principal interval used by SnakeLite."""
    return z - torch.pi * torch.round(z / torch.pi)


def _resolve_snake_lite_taylor_degree(snake_lite_taylor_degree: int | None) -> int:
    degree = 8 if snake_lite_taylor_degree is None else int(snake_lite_taylor_degree)
    if degree < 2 or degree % 2 != 0:
        raise ValueError(
            "snake_lite_taylor_degree must be an even integer >= 2. "
            f"Got {snake_lite_taylor_degree!r}."
        )
    return degree


@lru_cache(maxsize=None)
def _snake_lite_taylor_coeffs(
    snake_lite_taylor_degree: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    degree = _resolve_snake_lite_taylor_degree(snake_lite_taylor_degree)
    coeffs = []
    deriv_coeffs = []
    for n in range(1, degree // 2 + 1):
        coeff = ((-1) ** (n + 1)) * (2 ** (2 * n - 1)) / math.factorial(2 * n)
        coeffs.append(coeff)
        deriv_coeffs.append(2 * n * coeff)
    return tuple(coeffs), tuple(deriv_coeffs)


def _snake_lite_taylor_sin2(
    z: torch.Tensor,
    snake_lite_taylor_degree: int = 8,
) -> torch.Tensor:
    """Truncated Taylor approximation of sin^2(z) up to the requested even degree."""
    coeffs, _ = _snake_lite_taylor_coeffs(snake_lite_taylor_degree)
    z2 = z * z
    poly = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        poly = coeff + z2 * poly
    return z2 * poly


def _snake_lite_taylor_sin2_derivative(
    z: torch.Tensor,
    snake_lite_taylor_degree: int = 8,
) -> torch.Tensor:
    """Derivative of the truncated SnakeLite sin^2 Taylor approximation."""
    _, deriv_coeffs = _snake_lite_taylor_coeffs(snake_lite_taylor_degree)
    z2 = z * z
    poly = deriv_coeffs[-1]
    for coeff in reversed(deriv_coeffs[:-1]):
        poly = coeff + z2 * poly
    return z * poly


class _SnakeLiteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        beta: torch.Tensor,
        epsilon: float,
        snake_lite_taylor_degree: int,
    ) -> torch.Tensor:
        wrapped = _wrap_snake_lite_argument(x * beta)
        periodic = _snake_lite_taylor_sin2(wrapped, snake_lite_taylor_degree)
        safe_beta = beta + epsilon

        ctx.save_for_backward(x, beta, wrapped)
        ctx.beta_shape = beta.shape
        ctx.epsilon = epsilon
        ctx.snake_lite_taylor_degree = snake_lite_taylor_degree

        return x + periodic / safe_beta

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, beta, wrapped = ctx.saved_tensors
        safe_beta = beta + ctx.epsilon

        periodic = _snake_lite_taylor_sin2(
            wrapped,
            ctx.snake_lite_taylor_degree,
        )
        periodic_prime = _snake_lite_taylor_sin2_derivative(
            wrapped,
            ctx.snake_lite_taylor_degree,
        )

        # Ignore round() derivative almost everywhere, matching the paper's wrapped approximation in practice.
        grad_x = grad_output * (1.0 + periodic_prime)
        grad_beta = grad_output * (periodic_prime * x / safe_beta - periodic / (safe_beta * safe_beta))
        grad_beta = _reduce_broadcast_gradient(grad_beta, ctx.beta_shape)

        return grad_x, grad_beta, None, None


def _snake_lite_forward(
    x: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float,
    snake_lite_taylor_degree: int = 8,
) -> torch.Tensor:
    return _SnakeLiteFunction.apply(
        x,
        beta,
        epsilon,
        _resolve_snake_lite_taylor_degree(snake_lite_taylor_degree),
    )


_SNAKE_LITE_TRITON_COEFF_CACHE: dict[tuple[str, int | None, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_snake_lite_triton_coeff_tensors(
    device: torch.device,
    snake_lite_taylor_degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index, int(snake_lite_taylor_degree))
    cached = _SNAKE_LITE_TRITON_COEFF_CACHE.get(key)
    if cached is not None:
        return cached

    coeffs, deriv_coeffs = _snake_lite_taylor_coeffs(snake_lite_taylor_degree)
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=device)
    deriv_coeffs_t = torch.tensor(deriv_coeffs, dtype=torch.float32, device=device)
    cached = (coeffs_t, deriv_coeffs_t)
    _SNAKE_LITE_TRITON_COEFF_CACHE[key] = cached
    return cached


def _snake_lite_triton_forward(
    x: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float,
    snake_lite_taylor_degree: int = 8,
) -> torch.Tensor:
    degree = _resolve_snake_lite_taylor_degree(snake_lite_taylor_degree)
    if not x.is_cuda:
        warnings.warn("SnakeLiteTriton requested on a non-CUDA tensor; falling back to SnakeLite.")
        return _snake_lite_forward(x, beta, epsilon, degree)
    if not is_triton_snake_lite_available() or SnakeLiteTritonFunction is None:
        warnings.warn("SnakeLiteTriton requested but Triton is unavailable; falling back to SnakeLite.")
        return _snake_lite_forward(x, beta, epsilon, degree)
    coeffs_t, deriv_coeffs_t = _get_snake_lite_triton_coeff_tensors(x.device, degree)
    return SnakeLiteTritonFunction.apply(
        x,
        beta,
        epsilon,
        coeffs_t,
        deriv_coeffs_t,
    )


def _apply_split_condition_conv1d(
    prenet: nn.Conv1d,
    condition,
    target_length: int,
) -> torch.Tensor:
    """Apply a 1x1 condition conv without materializing repeated global condition tensors."""
    if not isinstance(condition, tuple):
        if condition.ndim == 2:
            condition = condition.unsqueeze(-1)
        if condition.shape[-1] != target_length:
            condition = torch.nn.functional.interpolate(condition, size=target_length, mode='nearest')
        return prenet(condition)

    global_condition, time_condition = condition
    global_dim = 0 if global_condition is None else int(global_condition.shape[1])
    time_dim = 0 if time_condition is None else int(time_condition.shape[1])
    expected_dim = int(prenet.in_channels)
    if global_dim + time_dim != expected_dim:
        raise ValueError(
            f"Condition split mismatch: expected {expected_dim} channels, "
            f"got global={global_dim}, time={time_dim}."
        )

    target_tensor = time_condition if time_condition is not None else global_condition
    weight = prenet.weight.to(device=target_tensor.device, dtype=target_tensor.dtype)
    bias = (
        prenet.bias.to(device=target_tensor.device, dtype=target_tensor.dtype)
        if prenet.bias is not None else None
    )
    out = None

    if time_condition is not None:
        if time_condition.ndim == 2:
            time_condition = time_condition.unsqueeze(-1)
        if time_condition.shape[-1] != target_length:
            time_condition = torch.nn.functional.interpolate(time_condition, size=target_length, mode='nearest')
        time_weight = weight[:, global_dim:, :]
        out = F.conv1d(time_condition, time_weight, bias=bias)
    elif bias is not None:
        out = bias.view(1, -1, 1).expand(-1, -1, target_length)

    if global_condition is not None:
        global_weight = weight[:, :global_dim, 0]
        global_out = F.linear(global_condition, global_weight).unsqueeze(-1)
        out = global_out if out is None else out + global_out

    if out is None:
        batch_size = 1 if global_condition is None else int(global_condition.shape[0])
        out = weight.new_zeros((batch_size, prenet.out_channels, target_length))

    return out


class Snake(nn.Module):
    '''
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        '''
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeLite(nn.Module):
    """
    SnakeLite from GenAE: a periodically wrapped Taylor approximation of Snake.

    The paper denotes the single periodic parameter as beta. This implementation
    keeps the Snake/SnakeBeta constructor arguments for drop-in compatibility
    with the existing codebase.
    """

    def __init__(
        self,
        in_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=False,
        snake_lite_taylor_degree: int = 8,
    ):
        super(SnakeLite, self).__init__()
        self.in_features = in_features
        self.snake_lite_taylor_degree = _resolve_snake_lite_taylor_degree(
            snake_lite_taylor_degree
        )

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        SnakeLite(x, beta) := x + P8(a(x, beta)) / beta
        where a(x, beta) = beta*x - pi*round(beta*x / pi).
        """
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)

        x = _snake_lite_forward(
            x,
            beta,
            self.no_div_by_zero,
            self.snake_lite_taylor_degree,
        )

        return x


class SnakeLiteTriton(SnakeLite):
    def forward(self, x):
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)

        return _snake_lite_triton_forward(
            x,
            beta,
            self.no_div_by_zero,
            self.snake_lite_taylor_degree,
        )


class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeLiteWithCondition(nn.Module):
    """
    SnakeLite with global conditioning.

    This conditioned extension is inferred from the repository's conditioned
    SnakeBeta variants because the paper only defines the unconditioned form.
    """

    def __init__(
        self,
        in_features,
        condition_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=False,
        snake_lite_taylor_degree: int = 8,
    ):
        super(SnakeLiteWithCondition, self).__init__()
        self.in_features = in_features
        self.snake_lite_taylor_degree = _resolve_snake_lite_taylor_degree(
            snake_lite_taylor_degree
        )

        self.condition_beta_prenet = torch.nn.Linear(condition_features, in_features)

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x, condition):
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)

        # Conservative modulation keeps the denominator away from zero.
        condition = torch.tanh(self.condition_beta_prenet(condition).unsqueeze(-1))
        beta = beta + 0.5 * condition

        x = _snake_lite_forward(
            x,
            beta,
            self.no_div_by_zero,
            self.snake_lite_taylor_degree,
        )

        return x


class SnakeLiteTritonWithCondition(SnakeLiteWithCondition):
    def forward(self, x, condition):
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)

        condition = torch.tanh(self.condition_beta_prenet(condition).unsqueeze(-1))
        beta = beta + 0.5 * condition

        return _snake_lite_triton_forward(
            x,
            beta,
            self.no_div_by_zero,
            self.snake_lite_taylor_degree,
        )


class SnakeBetaWithCondition(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Condition: (B, D), where D-dimension will be mapped to C dimensions
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
        - condition_alpha_prenet - trainable parameter that controls alpha and beta using condition
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256, 128)
        >>> x = torch.randn(256)
        >>> cond = torch.randn(128)
        >>> x = a1(x, cond)
    '''
    def __init__(self, in_features, condition_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: dimension of the input
            - condition_features: dimension of the condition vectors
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha, beta will be trained along with the rest of your model.
        '''
        super(SnakeBetaWithCondition, self).__init__()
        self.in_features = in_features
        
        self.condition_alpha_prenet = torch.nn.Linear(condition_features, in_features)
        # self.condition_beta_prenet = torch.nn.Linear(condition_features, in_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x, condition):
        '''
        condition: [B, D]
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        
        condition = torch.tanh(self.condition_alpha_prenet(condition).unsqueeze(-1))  # Same prenet for both alpha and beta, to save parameters
        alpha = alpha + condition
        beta = beta + 0.5 * condition  # multiply 0.5 for avoiding beta being too small
        
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeLiteWithTimeVaryingCondition(nn.Module):
    """
    SnakeLite with time-varying conditioning.

    This conditioned extension is inferred from the repository's conditioned
    SnakeBeta variants because the paper only defines the unconditioned form.
    """

    def __init__(
        self,
        in_features,
        condition_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=False,
        snake_lite_taylor_degree: int = 8,
    ):
        super(SnakeLiteWithTimeVaryingCondition, self).__init__()
        self.in_features = in_features
        self.snake_lite_taylor_degree = _resolve_snake_lite_taylor_degree(
            snake_lite_taylor_degree
        )

        self.condition_beta_prenet = torch.nn.Conv1d(condition_features, in_features, kernel_size=1)

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x, condition):
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)

        condition = torch.tanh(
            _apply_split_condition_conv1d(
                self.condition_beta_prenet,
                condition,
                target_length=x.shape[-1],
            )
        )
        beta = beta + 0.5 * condition

        x = _snake_lite_forward(
            x,
            beta,
            self.no_div_by_zero,
            self.snake_lite_taylor_degree,
        )

        return x


class SnakeLiteTritonWithTimeVaryingCondition(SnakeLiteWithTimeVaryingCondition):
    def forward(self, x, condition):
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)

        condition = torch.tanh(
            _apply_split_condition_conv1d(
                self.condition_beta_prenet,
                condition,
                target_length=x.shape[-1],
            )
        )
        beta = beta + 0.5 * condition

        return _snake_lite_triton_forward(
            x,
            beta,
            self.no_div_by_zero,
            self.snake_lite_taylor_degree,
        )


class SnakeBetaWithTimeVaryingCondition(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    with time-varying condition support
    Shape:
        - Input: (B, C, T)
        - Condition: (B, D, T), where D-dimension will be mapped to C dimensions
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
        - condition_alpha_prenet - trainable parameter that controls alpha and beta using condition
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = SnakeBetaWithTimeVaryingCondition(256, 128)
        >>> x = torch.randn(8, 256, 1000)  # (B, C, T)
        >>> cond = torch.randn(8, 128, 1000)  # (B, D, T)
        >>> x = a1(x, cond)
    '''
    def __init__(self, in_features, condition_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: dimension of the input
            - condition_features: dimension of the condition vectors
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha, beta will be trained along with the rest of your model.
        '''
        super(SnakeBetaWithTimeVaryingCondition, self).__init__()
        self.in_features = in_features
        
        # 1D Conv for time-varying condition processing
        self.condition_alpha_prenet = torch.nn.Conv1d(condition_features, in_features, kernel_size=1)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x, condition):
        '''
        x: [B, C, T]
        condition: [B, D, T]
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        
        condition = torch.tanh(
            _apply_split_condition_conv1d(
                self.condition_alpha_prenet,
                condition,
                target_length=x.shape[-1],
            )
        )
        
        # Apply time-varying modulation
        alpha = alpha + condition
        beta = beta + 0.5 * condition  # multiply 0.5 for avoiding beta being too small
        
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
