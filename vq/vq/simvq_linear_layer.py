import torch
import torch.nn as nn
from torch import Tensor

class OrthoIsoLinear(nn.Module):
    """
    y = s * x @ O^T  (O: orthogonal from skew-symmetric S via Cayley; s: isotropic scale > 0)
    - 자유도: d(d-1)/2 (S의 상삼각) + 1 (스케일)
    - 직교 변환(회전/반사) 외 전단/비균일 스케일 금지 → 코드북의 상대적 구조 보존에 유리
    """
    def __init__(self, dim: int, init_scale: float = 1.0, ema_decay: float = 0.0):
        super().__init__()
        self.dim = dim
        # unconstrained parameter -> skew-symmetric로 사용
        self.S = nn.Parameter(torch.zeros(dim, dim))
        # 등방 스케일 (양수 보장 위해 log 파라미터화)
        self.log_s = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        
        # EMA 사용 여부 설정
        self.use_ema = ema_decay > 0.0
        self.ema_decay = ema_decay
        
        # EMA를 사용할 경우에만 버퍼 등록
        if self.use_ema:
            self.register_buffer('S_ema', torch.zeros(dim, dim))
            self.register_buffer('log_s_ema', torch.log(torch.tensor(init_scale, dtype=torch.float32)))
            self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
        
        print(f"OrthoIsoLinear: dim={dim}, init_scale={init_scale}, ema_decay={ema_decay if self.use_ema else 0.0}")
        
    def update_ema(self):
        if not self.use_ema:
            return
            
        with torch.no_grad():
            if not self.initialized:
                self.S_ema.copy_(self.S)
                self.log_s_ema.copy_(self.log_s)
                self.initialized = torch.tensor(True, dtype=torch.bool)
            else:
                self.S_ema.mul_(self.ema_decay).add_(self.S * (1 - self.ema_decay))
                self.log_s_ema.mul_(self.ema_decay).add_(self.log_s * (1 - self.ema_decay))

    def _orthogonal(self) -> Tensor:
        # EMA 사용 여부에 따라 파라미터 선택
        param = self.S_ema if (not self.training and self.use_ema) else self.S
        
        # skew-symmetric로 투영
        S = param - param.transpose(0, 1)
        I = torch.eye(self.dim, device=S.device, dtype=S.dtype)
        # Cayley transform: O = (I - S)(I + S)^{-1}
        # (I + S)가 singular에 가까우면 안정성 위해 solve 사용
        O = torch.linalg.solve(I + S, I - S)
        return O

    @property
    def weight(self) -> Tensor:
        # 반환: W = s * O
        return self.scale * self._orthogonal()

    @property
    def scale(self) -> Tensor:
        # EMA 사용 여부에 따라 파라미터 선택
        param = self.log_s_ema if (not self.training and self.use_ema) else self.log_s
        return param.exp()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (..., dim)
        반환: (..., dim)  (완전 연결층처럼 동작; bias 없음)
        """
        # EMA 업데이트 (학습 모드이고 EMA 사용 시에만)
        if self.training and self.use_ema:
            self.update_ema()
            
        O = self._orthogonal()
        s = self.scale
        return (x @ O.transpose(-1, -2)) * s

    def extra_repr(self) -> str:
        if self.use_ema:
            return f"dim={self.dim}, scale={self.scale.item():.4f}, ema_decay={self.ema_decay:.4f}"
        else:
            return f"dim={self.dim}, scale={self.scale.item():.4f}, ema=False"

class LowRankMulLinear(nn.Module):
    r"""
    y = x @ W^T,  where  W = I + eps * (A @ B^T),   rank(W - I) <= r

    - 목적: codebook에 "전역적인 저차원 보정"만 허용 (전단/비균일 스케일처럼 과도한 왜곡 방지)
    - 항등 시작: A,B ≈ 0  ->  W ≈ I
    - eps: 보정 강도 (고정값 또는 학습 가능)
    - EMA: 평가시 더 안정적인 가중치 사용 옵션
    - 선택(normalize_update): AB^T 를 Frobenius-norm 1로 정규화 후 eps만으로 강도 조절
    """
    def __init__(
        self,
        dim: int,
        rank: int = 8,
        init_eps: float = 0.0,          # 0이면 완전 항등에서 시작
        ema_decay: float = 0.0,         # >0이면 EMA 사용
        learnable_eps: bool = True,     # True면 eps 학습
        normalize_update: bool = True,  # AB^T 크기 정규화 여부
        init_std: float = 1e-3          # A,B 초기화 표준편차(0으로 두면 완전 항등)
    ):
        super().__init__()
        assert rank > 0 and rank <= dim, "rank must be in [1, dim]"
        self.dim = dim
        self.rank = rank
        self.normalize_update = normalize_update

        # A, B: [d, r]
        self.A = nn.Parameter(torch.zeros(dim, rank))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        if init_std > 0:
            nn.init.normal_(self.A, std=init_std)
            nn.init.normal_(self.B, std=init_std)

        # eps: 보정 세기 (log-파라미터화로 양/음 모두 가능하게 하려면 그냥 선형)
        # 여기서는 안정적으로 양수 스케일을 원하면 exp(log_eps)로, 부호도 허용하려면 eps 자체 파라미터 사용
        if learnable_eps:
            self.log_eps = nn.Parameter(torch.tensor(init_eps).log() if init_eps > 0 else torch.tensor(-float("inf")))
        else:
            self.register_buffer("fixed_eps", torch.tensor(init_eps, dtype=torch.float32))
            self.log_eps = None

        # EMA 설정
        self.use_ema = ema_decay > 0.0
        self.ema_decay = ema_decay
        if self.use_ema:
            self.register_buffer("A_ema", self.A.detach().clone())
            self.register_buffer("B_ema", self.B.detach().clone())
            if self.log_eps is not None:
                self.register_buffer("log_eps_ema", self.log_eps.detach().clone())
            else:
                self.register_buffer("log_eps_ema", torch.tensor(float("nan")))
            self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    # ---------------- EMA ----------------
    def update_ema(self):
        if not self.use_ema:
            return
        with torch.no_grad():
            if not self.initialized:
                self.A_ema.copy_(self.A)
                self.B_ema.copy_(self.B)
                if self.log_eps is not None:
                    self.log_eps_ema.copy_(self.log_eps)
                self.initialized = torch.tensor(True, dtype=torch.bool)
            else:
                d = self.ema_decay
                self.A_ema.mul_(d).add_(self.A * (1 - d))
                self.B_ema.mul_(d).add_(self.B * (1 - d))
                if self.log_eps is not None:
                    self.log_eps_ema.mul_(d).add_(self.log_eps * (1 - d))

    # -------------- Core weight --------------
    def _core_params(self):
        """훈련/평가 & EMA 사용 여부에 따라 사용할 (A,B,eps) 선택"""
        if (not self.training) and self.use_ema:
            A = self.A_ema
            B = self.B_ema
            if self.log_eps is not None:
                eps = self.log_eps_ema.exp().clamp(min=0.0)  # learnable_eps의 EMA
            else:
                eps = self.fixed_eps
        else:
            A = self.A
            B = self.B
            if self.log_eps is not None:
                eps = self.log_eps.exp().clamp(min=0.0)
            else:
                eps = self.fixed_eps
        return A, B, eps

    def _weight(self) -> Tensor:
        """
        W = I + eps * (A @ B^T)
        필요 시 AB^T를 Fro-norm 1로 정규화하여 eps만으로 강도 조절.
        """
        A, B, eps = self._core_params()
        # [d, r] @ [r, d] = [d, d]  (하지만 실제 곱셈은 메모리 고려해 matmul)
        update = A @ B.transpose(0, 1)  # dxd

        if self.normalize_update:
            # 안정적 정규화 (Frobenius norm, eps는 보정 강도)
            norm = update.norm(p="fro").clamp_min(1e-12)
            update = update / norm

        I = torch.eye(self.dim, device=update.device, dtype=update.dtype)
        W = I + eps * update
        return W

    @property
    def weight(self) -> Tensor:
        return self._weight()

    # -------------- Forward --------------
    def forward(self, x: Tensor) -> Tensor:
        """
        x: (..., dim)
        return: (..., dim)
        """
        if self.training and self.use_ema:
            self.update_ema()
        W = self._weight()               # [d, d]
        return x @ W.transpose(-1, -2)

    # -------------- Utils --------------
    def extra_repr(self) -> str:
        if self.log_eps is None:
            eps_str = f"{float(self.fixed_eps):.4e}"
        else:
            eps_val = self.log_eps.detach().exp().item() if torch.isfinite(self.log_eps.detach()) else 0.0
            eps_str = f"{eps_val:.4e} (learnable)"
        ema_str = f", ema_decay={self.ema_decay:.4f}" if self.use_ema else ", ema=False"
        return f"dim={self.dim}, rank={self.rank}, eps={eps_str}, normalize_update={self.normalize_update}{ema_str}"

    # 선택: 정규화/모니터링용 헬퍼 (loss에 더하기 좋음)
    def regularization_terms(self):
        """
        반환값:
          - fro_norm_update: ||A B^T||_F (정규화 전에 측정)
          - nuclear_norm_approx: ||A||_F * ||B||_F (상계로 사용 가능)
        """
        with torch.no_grad():
            fro = (self.A @ self.B.T).norm(p="fro")
            nu_ub = self.A.norm(p="fro") * self.B.norm(p="fro")
        return fro, nu_ub
