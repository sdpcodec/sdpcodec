import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

def split_gates(gates: torch.Tensor, hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return gates.chunk(4, dim=-1)

class LSTMCellPreact(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    def forward(self, x_t: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hx
        preact = self.Wx(x_t) + self.Wh(h_prev)
        i, f, g, o = split_gates(preact, self.hidden_size)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        g = torch.tanh(g);    o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
    def preact_only(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return self.Wx(x_t) + self.Wh(h_prev)

class HyperLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cond_size: int,
        *,
        bias: bool = True,
        nonlinear: str = "tanh",
        hyper: str = "static",
        hyper_hidden: int = 128,
        feed_x_into_hyper: bool = False,
        residual: bool = False,
        residual_scale: float = 1.0,
        use_residual_proj_if_mismatch: bool = True,
        bidirectional: bool = False,
        merge: str = "cat",
        num_layers: int = 1,
    ):
        super().__init__()
        assert hyper in ("static", "dynamic")
        assert merge in ("cat", "sum", "proj")
        assert num_layers >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.nonlinear = nonlinear
        self.hyper = hyper
        self.hyper_hidden = hyper_hidden
        self.feed_x_into_hyper = feed_x_into_hyper
        self.residual = residual
        self.residual_scale = residual_scale
        self.use_residual_proj_if_mismatch = use_residual_proj_if_mismatch
        self.bidirectional = bidirectional
        self.merge = merge
        self.num_layers = num_layers

        if not bidirectional:
            self.layers = nn.ModuleList()
            in_size = input_size
            for _ in range(num_layers):
                self.layers.append(self._build_direction(bias, in_size))
                in_size = hidden_size
            self.merge_proj = None
        else:
            self.fwd_layers = nn.ModuleList()
            self.bwd_layers = nn.ModuleList()
            self.merge_proj = nn.ModuleList() if merge == "proj" else None
            in_size = input_size
            for _ in range(num_layers):
                self.fwd_layers.append(self._build_direction(bias, in_size))
                self.bwd_layers.append(self._build_direction(bias, in_size))
                if merge == "proj":
                    proj = nn.Linear(2 * hidden_size, hidden_size)
                    nn.init.xavier_uniform_(proj.weight); nn.init.zeros_(proj.bias)
                    self.merge_proj.append(proj)
                if merge == "cat":
                    in_size = 2 * hidden_size
                else:
                    in_size = hidden_size

    class _Dir(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            cond_size: int,
            *,
            bias: bool,
            nonlinear: str,
            hyper: str,
            hyper_hidden: int,
            feed_x_into_hyper: bool,
            residual: bool,
            residual_scale: float,
            use_residual_proj_if_mismatch: bool,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.cond_size = cond_size
            self.hyper = hyper
            self.feed_x_into_hyper = feed_x_into_hyper
            self.residual = residual
            self.residual_scale = residual_scale

            self.cell = LSTMCellPreact(input_size, hidden_size, bias=bias)

            if hyper == "static":
                self.to_gamma_beta = nn.Linear(cond_size, 2 * 4 * hidden_size)
                nn.init.xavier_uniform_(self.to_gamma_beta.weight)
                nn.init.zeros_(self.to_gamma_beta.bias)
            else:
                self.to_gamma_beta = None

            if hyper == "dynamic":
                hyper_in = cond_size + hidden_size + (input_size if feed_x_into_hyper else 0)
                self.hyper_rnn = nn.GRU(input_size=hyper_in, hidden_size=hyper_hidden, batch_first=True)
                self.hyper_to_gb = nn.Linear(hyper_hidden, 2 * 4 * hidden_size)
                nn.init.xavier_uniform_(self.hyper_to_gb.weight)
                nn.init.zeros_(self.hyper_to_gb.bias)
                for name, p in self.hyper_rnn.named_parameters():
                    if 'weight' in name: nn.init.xavier_uniform_(p)
                    elif 'bias' in name: nn.init.zeros_(p)
            else:
                self.hyper_rnn = None
                self.hyper_to_gb = None

            need_proj = (input_size != hidden_size)
            self.res_proj = None
            if residual and need_proj and use_residual_proj_if_mismatch:
                self.res_proj = nn.Linear(input_size, hidden_size, bias=False)
                nn.init.xavier_uniform_(self.res_proj.weight)

            if nonlinear == "tanh":
                self._nonlin = torch.tanh
            elif nonlinear == "sigmoid":
                self._nonlin = torch.sigmoid
            else:
                self._nonlin = None

        def forward(
            self,
            x: torch.Tensor,  # (B,T,D)
            z: torch.Tensor,  # (B,C) or (B,T,C)
            hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            B, T, Din = x.shape
            H = self.hidden_size
            device = x.device
            dtype = x.dtype

            if hx is None:
                h_t = torch.zeros(B, H, device=device, dtype=dtype)
                c_t = torch.zeros(B, H, device=device, dtype=dtype)
            else:
                h_t, c_t = hx

            pre_x = self.cell.Wx(x.reshape(B * T, Din)).reshape(B, T, 4 * H)

            if self.residual:
                if self.res_proj is None:
                    res = x if Din == H else None
                else:
                    res = self.res_proj(x.reshape(B * T, Din)).reshape(B, T, H)
            else:
                res = None

            # z shape: (B,C) or (B,T,C)
            z_is_time = (z.dim() == 3 and z.shape[1] == T)
            if self.hyper == "static":
                if z_is_time:
                    gb = self.to_gamma_beta(z.reshape(B * T, -1)).reshape(B, T, 8 * H)
                    gamma, beta = gb.chunk(2, dim=-1)
                    if self._nonlin is not None:
                        gamma = 1.0 + 0.5 * self._nonlin(gamma)
                        beta  = 0.5 * self._nonlin(beta)
                else:
                    gb = self.to_gamma_beta(z)  # (B,8H)
                    gamma, beta = gb.chunk(2, dim=-1)
                    if self._nonlin is not None:
                        gamma = 1.0 + 0.5 * self._nonlin(gamma)
                        beta  = 0.5 * self._nonlin(beta)
            else:
                h_hyper = torch.zeros(1, B, self.hyper_rnn.hidden_size, device=device, dtype=dtype)

            outs = []
            Wh = self.cell.Wh
            for t in range(T):
                pre = pre_x[:, t, :] + Wh(h_t)
                if self.hyper == "static":
                    if z_is_time:
                        gamma_t = gamma[:, t, :]
                        beta_t = beta[:, t, :]
                    else:
                        gamma_t = gamma
                        beta_t = beta
                    pre = gamma_t * pre + beta_t
                else:
                    x_t = x[:, t, :]
                    if z_is_time:
                        z_t = z[:, t, :]
                    else:
                        z_t = z
                    if self.feed_x_into_hyper:
                        hyper_in_t = torch.cat([z_t, h_t, x_t], dim=-1)
                    else:
                        hyper_in_t = torch.cat([z_t, h_t], dim=-1)
                    hyper_out, h_hyper = self.hyper_rnn(hyper_in_t.unsqueeze(1), h_hyper)
                    gb_t = self.hyper_to_gb(hyper_out.squeeze(1))
                    gamma_t, beta_t = gb_t.chunk(2, dim=-1)
                    gamma_t = 1.0 + 0.5 * torch.tanh(gamma_t)
                    beta_t  = 0.5 * torch.tanh(beta_t)
                    pre = gamma_t * pre + beta_t

                i, f, g, o = split_gates(pre, H)
                i = torch.sigmoid(i); f = torch.sigmoid(f)
                g = torch.tanh(g);    o = torch.sigmoid(o)
                c_t = f * c_t + i * g
                h_t = o * torch.tanh(c_t)
                outs.append(h_t.unsqueeze(1))

            y = torch.cat(outs, dim=1)
            if res is not None and self.residual_scale != 0.0:
                y = y + self.residual_scale * res
            return y, (h_t, c_t)

    def _build_direction(self, bias: bool, in_size: int) -> nn.Module:
        return HyperLSTM._Dir(
            in_size, self.hidden_size, self.cond_size,
            bias=bias,
            nonlinear=self.nonlinear,
            hyper=self.hyper,
            hyper_hidden=self.hyper_hidden,
            feed_x_into_hyper=self.feed_x_into_hyper,
            residual=self.residual,
            residual_scale=self.residual_scale,
            use_residual_proj_if_mismatch=self.use_residual_proj_if_mismatch,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B,T,D0)
        z: torch.Tensor,  # (B,C) or (B,T,C)
        hx: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            ]
        ] = None,
    ):
        if not self.bidirectional:
            x_l = x
            h_l = c_l = None
            for li in range(self.num_layers):
                init_l = hx if (li == 0 and isinstance(hx, tuple) and len(hx) == 2) else None
                y_l, (h_l, c_l) = self.layers[li](x_l, z, init_l)
                x_l = y_l
            return x_l, (h_l, c_l)

        x_l = x
        hf_l = cf_l = hb_l = cb_l = None
        for li in range(self.num_layers):
            hx_f = hx[0] if (li == 0 and hx is not None) else None
            hx_b = hx[1] if (li == 0 and hx is not None) else None
            y_f, (hf_l, cf_l) = self.fwd_layers[li](x_l, z, hx_f)
            x_rev = torch.flip(x_l, dims=[1])
            y_b_rev, (hb_l, cb_l) = self.bwd_layers[li](x_rev, z, hx_b)
            y_b = torch.flip(y_b_rev, dims=[1])
            if self.merge == "cat":
                y = torch.cat([y_f, y_b], dim=-1)
            elif self.merge == "sum":
                y = y_f + y_b
            else:
                y = self.merge_proj[li](torch.cat([y_f, y_b], -1))
            x_l = y
        return x_l, ((hf_l, cf_l), (hb_l, cb_l))

if __name__ == "__main__":
    B, T, Din, H, Dz = 2, 16, 32, 64, 8
    x = torch.randn(B, T, Din)
    z = torch.randn(B, Dz)
    zt = torch.randn(B, T, Dz)

    print("== HyperLSTM (static, uni, 1 layer, global z) ==")
    m1 = HyperLSTM(Din, H, Dz, hyper="static", residual=False, bidirectional=False, num_layers=1)
    y1, (h1, c1) = m1(x, z)
    print(y1.shape, h1.shape, c1.shape)

    print("== HyperLSTM (static, uni, 1 layer, time-varying z) ==")
    y1t, (h1t, c1t) = m1(x, zt)
    print(y1t.shape, h1t.shape, c1t.shape)

    print("== HyperLSTM (dynamic, uni, 3 layers, time-varying z) ==")
    m2 = HyperLSTM(Din, H, Dz, hyper="dynamic", hyper_hidden=64, feed_x_into_hyper=True, bidirectional=False, num_layers=3)
    y2, (h2, c2) = m2(x, zt)
    print(y2.shape, h2.shape, c2.shape)

    print("== HyperLSTM (static+residual, bi, proj, 2 layers, time-varying z) ==")
    m3 = HyperLSTM(Din, H, Dz, hyper="static", residual=True, residual_scale=1.0, bidirectional=True, merge="proj", num_layers=2)
    y3, ((hf, cf), (hb, cb)) = m3(x, zt)
    print(y3.shape, hf.shape, cf.shape, hb.shape, cb.shape)