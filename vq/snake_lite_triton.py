import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False
    TRITON_IMPORT_ERROR = exc


if TRITON_AVAILABLE:
    @triton.jit
    def _snake_lite_horner_z2(
        z2,
        coeff_ptr,
        NUM_TERMS: tl.constexpr,
    ):
        coeff = z2 * 0.0 + tl.load(coeff_ptr + (NUM_TERMS - 1)).to(tl.float32)
        for i in range(NUM_TERMS - 2, -1, -1):
            coeff = tl.load(coeff_ptr + i).to(tl.float32) + z2 * coeff
        return z2 * coeff


    @triton.jit
    def _snake_lite_horner_z(
        z,
        z2,
        coeff_ptr,
        NUM_TERMS: tl.constexpr,
    ):
        coeff = z2 * 0.0 + tl.load(coeff_ptr + (NUM_TERMS - 1)).to(tl.float32)
        for i in range(NUM_TERMS - 2, -1, -1):
            coeff = tl.load(coeff_ptr + i).to(tl.float32) + z2 * coeff
        return z * coeff


    @triton.jit
    def _snake_lite_forward_kernel(
        x_ptr,
        beta_ptr,
        coeff_ptr,
        out_ptr,
        n_elements,
        epsilon,
        NUM_TERMS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        pi = 3.141592653589793
        q = (x * beta) / pi
        rounded = tl.where(q >= 0.0, tl.floor(q + 0.5), -tl.floor(-q + 0.5))
        wrapped = x * beta - pi * rounded

        z2 = wrapped * wrapped
        periodic = _snake_lite_horner_z2(z2, coeff_ptr, NUM_TERMS)

        out = x + periodic / (beta + epsilon)
        tl.store(out_ptr + offsets, out, mask=mask)


    @triton.jit
    def _snake_lite_backward_kernel(
        grad_output_ptr,
        x_ptr,
        beta_ptr,
        coeff_ptr,
        deriv_coeff_ptr,
        grad_x_ptr,
        grad_beta_ptr,
        n_elements,
        epsilon,
        NUM_TERMS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        pi = 3.141592653589793
        q = (x * beta) / pi
        rounded = tl.where(q >= 0.0, tl.floor(q + 0.5), -tl.floor(-q + 0.5))
        wrapped = x * beta - pi * rounded

        z2 = wrapped * wrapped
        periodic = _snake_lite_horner_z2(z2, coeff_ptr, NUM_TERMS)
        periodic_prime = _snake_lite_horner_z(
            wrapped,
            z2,
            deriv_coeff_ptr,
            NUM_TERMS,
        )

        safe_beta = beta + epsilon
        grad_x = grad_output * (1.0 + periodic_prime)
        grad_beta = grad_output * ((periodic_prime * x) / safe_beta - periodic / (safe_beta * safe_beta))

        tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
        tl.store(grad_beta_ptr + offsets, grad_beta, mask=mask)


def is_triton_snake_lite_available() -> bool:
    return TRITON_AVAILABLE


class SnakeLiteTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        beta: torch.Tensor,
        epsilon: float,
        coeffs: torch.Tensor,
        deriv_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
        if not x.is_cuda:
            raise RuntimeError("SnakeLiteTriton requires CUDA tensors.")

        x_contig = x.contiguous()
        beta_expanded = beta.expand_as(x_contig).contiguous()
        coeffs_contig = coeffs.contiguous()
        deriv_coeffs_contig = deriv_coeffs.contiguous()
        out = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        num_terms = int(coeffs_contig.numel())

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _snake_lite_forward_kernel[grid](
            x_contig,
            beta_expanded,
            coeffs_contig,
            out,
            n_elements,
            epsilon,
            NUM_TERMS=num_terms,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

        ctx.save_for_backward(x_contig, beta, coeffs_contig, deriv_coeffs_contig)
        ctx.epsilon = epsilon
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, beta, coeffs, deriv_coeffs = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        beta_expanded = beta.expand_as(x).contiguous()
        grad_x = torch.empty_like(x)
        grad_beta_expanded = torch.empty_like(x)
        n_elements = x.numel()
        num_terms = int(coeffs.numel())

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _snake_lite_backward_kernel[grid](
            grad_output,
            x,
            beta_expanded,
            coeffs,
            deriv_coeffs,
            grad_x,
            grad_beta_expanded,
            n_elements,
            ctx.epsilon,
            NUM_TERMS=num_terms,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

        grad_beta = grad_beta_expanded.sum_to_size(beta.shape)
        return grad_x, grad_beta, None, None, None
