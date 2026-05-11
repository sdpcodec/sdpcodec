'''
Brought from https://github.com/youngsheen/SimVQ/tree/main
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any
from torch import einsum
from einops import rearrange
from collections import namedtuple
import math
from termcolor import colored
from s3prl.nn import S3PRLUpstream
from s3prl.upstream.wav2vec.wav2vec_model import KmeansVectorQuantizer
from vq.vq.simvq_linear_layer import OrthoIsoLinear

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'codebook_entropy', 'commitment', 'avg_probs'])

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    # FP16-safe EMA 업데이트
    if new.dtype in (torch.float16, torch.bfloat16):
        new = new.float()
    moving_avg.data.mul_(decay).add_(new, alpha=(1.0 - decay))

class SimVQ(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, codebook_size, codebook_dim, commitment=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, threshold_ema_dead_code=0.2, decay=0.99,
                 ):
        super().__init__()
        self.beta = commitment
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.e_dim = codebook_dim

        self.n_e = codebook_size
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)
        print(colored(f'SimVQ: Randomly initialized codebook with {self.n_e} entries and {self.e_dim} dimensions.', 'green', attrs=['bold']))
        

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. Using {self.unknown_index} for unknown indices.")
        else:
            # 실제 코드북 크기와 일치시킴 (S3Tokenizer 사용 시 중요)
            self.re_embed = self.n_e

        # EMA용 코드 사용량 버퍼 (re_embed 크기)
        self.register_buffer("cluster_size", torch.zeros(self.re_embed))

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape, device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        quant_codebook = self.embedding_proj(self.embedding.weight)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        # compute loss for embedding
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # straight-through
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # (2D 버전은 코드 사용량 집계/동기화 없음)
        return (z_q, torch.tensor(0.0, device=z_q.device), min_encoding_indices), \
               LossBreakdown(torch.tensor(0.0, device=z_q.device),
                             torch.tensor(0.0, device=z_q.device),
                             commit_loss,
                             torch.tensor(0.0, device=z_q.device))

class SimVQ1D(SimVQ):

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def obtain_min_indices(self, z_flattened):
        self.quant_codebook = self.embedding_proj(self.embedding.weight)  # [N, D]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.quant_codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.quant_codebook, 'n d -> d n'))
        return torch.argmin(d, dim=1)
    
    def obtain_embedding_from_indices(self, z, min_encoding_indices):
        return F.embedding(min_encoding_indices, self.quant_codebook).view(z.shape)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 3 and indices.size(1) == 1:
            indices = indices.squeeze(1)
        if indices.dim() == 3 and indices.size(-1) == 1:
            indices = indices.squeeze(-1)
        if indices.dim() != 2:
            raise ValueError(f"SimVQ1D indices must be [B, T], got shape {tuple(indices.shape)}")

        quant_codebook = self.embedding_proj(self.embedding.weight)
        z_q = F.embedding(indices.long(), quant_codebook)
        return rearrange(z_q, 'b h c -> b c h').contiguous()

    def forward(self, z, total_step=None, temp=None, rescale_logits=False, return_logits=False, produce_targets=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"

        # z: [B, C, T] -> [B, T, C]
        z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim

        z_flattened = z.view(-1, self.e_dim)

        min_encoding_indices = self.obtain_min_indices(z_flattened)
        # print("Quantized indices:", min_encoding_indices)

        z_q = self.obtain_embedding_from_indices(z, min_encoding_indices)

        # loss
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # straight-through
        z_q = z + (z_q - z).detach()

        # back to [B, C, H]
        z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        # remap 처리
        if self.remap is not None:
            tmp = min_encoding_indices.reshape(z.shape[0], -1)          # [B, H]
            tmp = self.remap_to_used(tmp)                                # [B, H] in [0..re_embed-1]
            min_encoding_indices = tmp.reshape(-1, 1)                    # [B*H, 1]

        # sane index shape for 1D: [B, H]
        if self.sane_index_shape:
            if min_encoding_indices.dim() == 2 and min_encoding_indices.size(-1) == 1:
                sane_inds = min_encoding_indices.view(z_q.shape[0], z_q.shape[-1])
            else:
                sane_inds = min_encoding_indices.view(z_q.shape[0], -1)
        else:
            sane_inds = None  # not used downstream

        # --- DDP-safe code usage stats ---
        b = z_q.shape[0]
        t = z_q.shape[-1]

        # reshape indices to [B, T]
        inds = min_encoding_indices
        if inds.dim() == 2 and inds.size(-1) == 1:
            inds = inds.view(b, t)
        elif inds.dim() == 1:
            inds = inds.view(b, t)
        else:
            inds = inds.view(b, -1)

        # total codebook size (constant across ranks ideally)
        num_codes_local = int(getattr(self, "re_embed", self.n_e))

        # 안전 체크: 인덱스가 코드북 범위를 넘지 않도록
        flat_inds = inds.reshape(-1).to(dtype=torch.long)
        if flat_inds.numel() > 0:
            max_ind = int(flat_inds.max().item())
            if max_ind >= num_codes_local:
                raise RuntimeError(f"indices out of range: max {max_ind} >= num_codes {num_codes_local}")

        # build fixed-length counts (float32 for stable reduce)
        counts = torch.bincount(flat_inds, minlength=num_codes_local).to(dtype=torch.float32)

        world = 1
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist:
            world = torch.distributed.get_world_size()

        with torch.no_grad():
            if is_dist and world > 1:
                device = counts.device

                # Check declared num_codes consistency across ranks
                len_tensor = torch.tensor([num_codes_local], device=device, dtype=torch.int64)
                len_max = len_tensor.clone()
                len_min = len_tensor.clone()
                torch.distributed.all_reduce(len_max, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(len_min, op=torch.distributed.ReduceOp.MIN)

                target_len = int(len_max.item())
                if target_len != int(len_min.item()):
                    # Recompute counts padded to max length across ranks
                    if counts.numel() != target_len:
                        counts = torch.bincount(flat_inds, minlength=target_len).to(dtype=torch.float32)

                # total tokens across ranks (float32)
                local_total = torch.tensor([float(flat_inds.numel())], device=device, dtype=torch.float32)
                global_total = local_total.clone()

                # Now safe: all ranks have counts.shape consistent
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(global_total, op=torch.distributed.ReduceOp.SUM)

                avg_probs = counts / (global_total.item() + 1e-8)
                active_counts = counts
            else:
                avg_probs = counts / float(flat_inds.numel() + 1e-8)
                active_counts = counts

            # Perplexity in float32
            eps = 1e-6
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))

            # cluster_size 길이 일치 보장
            if self.cluster_size.numel() != self.re_embed:
                self.cluster_size = torch.zeros(self.re_embed, device=z_q.device, dtype=torch.float32)

            # EMA update of cluster_size (use training mode)
            if self.training:
                ema_inplace(self.cluster_size, active_counts[: self.re_embed], self.decay)

            active_num = (self.cluster_size > self.threshold_ema_dead_code).sum().float()

        # 반환 형태: (z_q, indices, vq_loss, perplexity, active_num)
        if self.sane_index_shape and sane_inds is not None:
            return z_q, sane_inds, commit_loss, perplexity, active_num
        else:
            return z_q, min_encoding_indices, commit_loss, perplexity, active_num

class KmeansVectorQuantizer_(KmeansVectorQuantizer):
    def __init__(self, *args: Any, use_vqw2v_embed: bool = True, freeze_embed: bool = True,
                 **kwargs: Any):
        # 1) 부모 초기화: 부모가 self.num_vars, self.groups, self.var_dim, self.embedding 등을 세팅
        super().__init__(*args, **kwargs)

        # 2) 필요 시 vq-wav2vec의 코드북으로 부모의 embedding 교체
        self.freeze_embed = freeze_embed
        self.use_vqw2v_embed = use_vqw2v_embed
        enc = S3PRLUpstream("vq_wav2vec_kmeans")
        with torch.no_grad():
            E = enc.upstream.model.vector_quantizer.embedding  # [K, gE, Dg]
            # 부모가 만든 self.embedding(Parameter)에 복사
            self.embedding.data.resize_(E.shape).copy_(E)
            if use_vqw2v_embed:
                print(colored(f'KmeansVectorQuantizer_: Using PRETRAINED VQ-Wav2Vec codebook with {E.shape[0]} entries and {E.shape[2]} dimensions.', 'green', attrs=['bold']))
            else:
                # assert freeze_embed == True, "If not using vq-wav2vec embed which means using additional discrete token embeddings, must freeze the codebook and projection."
                if self.combine_groups:
                    print(colored(f'KmeansVectorQuantizer_: RANDOM codebook (shared across {self.groups} groups): {self.num_vars} x {self.var_dim}', 'green', attrs=['bold']))
                    self.discrete_token_embeddings = nn.Embedding(self.num_vars, self.var_dim)
                else:
                    print(colored(f'KmeansVectorQuantizer_: RANDOM codebooks (per-group flattened): ({self.num_vars} * {self.groups}) x {self.var_dim}', 'green', attrs=['bold']))
                    self.discrete_token_embeddings = nn.Embedding(self.num_vars * self.groups, self.var_dim)
                nn.init.normal_(self.discrete_token_embeddings.weight, mean=0.0, std=self.var_dim ** -0.5)
            self.projection = enc.upstream.model.vector_quantizer.projection

            self.embedding.requires_grad = not freeze_embed
            self.projection.requires_grad = not freeze_embed

        del enc
    
    def forward(self, x, total_step=None, produce_targets=False):
        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape

        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (
            (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1))
            .view(self.num_vars, bsz, tsz, self.groups, -1)
            .norm(dim=-1, p=2)
        )
        idx = d.argmin(dim=0)

        if self.use_vqw2v_embed:
            zq = (
                torch.stack(
                    [
                        self.expand_embedding[idx[..., group], group]
                        for group in range(self.groups)
                    ],
                    dim=-2,
                )
                .view(bsz, tsz, self.groups * self.var_dim)
                .permute(0, 2, 1)
            )
        else:
            if self.combine_groups:
                # 공유 코드북: idx ∈ [0,num_vars-1]
                emb = self.discrete_token_embeddings(idx)              # [B,T,G] -> [B,T,G,D]
            else:
                # 그룹별 분리 코드북: 전역 인덱스 = idx + g*K
                group_offsets = torch.arange(self.groups, device=idx.device).view(1,1,self.groups) * self.num_vars  # [1,1,G]
                flat_idx = idx + group_offsets                           # [B,T,G] in [0, num_vars*groups-1]
                emb = self.discrete_token_embeddings(flat_idx)           # [B,T,G,D]
            zq = emb.permute(0, 2, 3, 1).contiguous().view(bsz, self.groups * self.var_dim, tsz)

        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        # x = self._pass_grad(ze, zq)
        x_q = zq

        hard_x = (
            idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
            .scatter_(-1, idx.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x_q = x_q.transpose(1, 2)  # BCT -> BTC

        result["x"] = x_q

        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        result["kmeans_loss"] = latent_loss + self.gamma * commitment_loss
        # print(f'x_q : {result["x"]}, idx : {idx}  ')

        # return result["x"].detach(), result.get("targets", None), result["kmeans_loss"], result["code_perplexity"], torch.tensor(0.0, device=x.device)

        # x: ST 경로 유지(encoder로 grad 전달). 지표는 detach해서 불필요한 그래프 전파 차단
        active_num = result["code_perplexity"].new_tensor(0.0)
        return (
            result["x"].detach(),                                  # grad 유지
            result.get("targets", None),
            torch.tensor(0.0, device=x_q.device) if self.freeze_embed else result["kmeans_loss"],                        # 손실은 grad 필요
            result["code_perplexity"].detach(),           # 로깅용, grad 불필요
            active_num.detach(),                          # 로깅/집계용
        )


class KmeansVectorQuantizer_SimVQ(KmeansVectorQuantizer_):
    def __init__(self, *args: Any, use_vqw2v_embed: bool = True, freeze_embed: bool = True, 
                 commitment: float = 0.25,
                 simvq_linear_layer_type: str = 'linear', ema_decay: float = 0.0,
                 **kwargs: Any,):
        # Initialize parent class
        super().__init__(*args, use_vqw2v_embed=use_vqw2v_embed, freeze_embed=freeze_embed, **kwargs)
        
        # SimVQ parameters
        self.beta = commitment
        
        # Create embedding projection layer
        K, gE, dg = self.embedding.shape
        if simvq_linear_layer_type == 'linear':
            print(colored(f'Using Linear for embedding projection.', 'red', attrs=['bold']))
            self.embedding_proj = nn.Linear(dg, dg)
        elif simvq_linear_layer_type == 'ortho_iso':
            print(colored(f'Using OrthoIsoLinear for embedding projection, ema_decay : {ema_decay}', 'red', attrs=['bold']))
            print(f'Using OrthoIsoLinear for embedding projection, ema_decay : {ema_decay}')
            self.embedding_proj = OrthoIsoLinear(dg, 1, ema_decay=ema_decay)
        elif simvq_linear_layer_type == 'lowrank_mul':
            from vq.vq.simvq_linear_layer import LowRankMulLinear
            rank = 4
            print(colored(f'Using LowRankMulLinear (rank={rank}) for embedding projection.', 'red', attrs=['bold']))
            self.embedding_proj = LowRankMulLinear(dg, rank=rank, eps=0.0)
        
        # self.projected_embedding = self.embedding_proj(self.embedding)

    @property
    def expand_projected_embedding(self):
        # Calculate projected_embedding on-the-fly to ensure device consistency
        projected_embedding = self.embedding_proj(self.embedding)
        if self.combine_groups:
            return projected_embedding.expand(self.num_vars, self.groups, self.var_dim)
        return projected_embedding

    def forward(self, x, total_step=None, produce_targets=False):
        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape

        # Project input features
        ze = self.projection(x)

        
        # Reshape for distance calculation
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        
        # Calculate distances using projected embeddings
        d = (
            (ze_.unsqueeze(0) - self.expand_projected_embedding.unsqueeze(1).unsqueeze(1))
            .view(self.num_vars, bsz, tsz, self.groups, -1)
            .norm(dim=-1, p=2)
        )
        
        # Get nearest neighbor indices
        idx = d.argmin(dim=0)
        
        # Get quantized vectors using projected embeddings
        zq = (
            torch.stack(
                [
                    self.expand_projected_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
            .view(bsz, tsz, self.groups * self.var_dim)
            .permute(0, 2, 1)
        )
        
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        
        # SimVQ commitment loss calculation
        commit_loss = self.beta * torch.mean((zq.detach() - ze) ** 2) + torch.mean((zq - ze.detach()) ** 2)
        
        # Straight-through estimator: pass gradients from ze to zq
        x = ze + (zq - ze).detach()
        
        # Compute perplexity metrics
        hard_x = (
            idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
            .scatter_(-1, idx.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
            
        # Return format to match SimVQ1D
        active_num = torch.tensor(0.0, device=x.device)
        return (
            x,                    # Keep gradients for backprop
            idx,                  # Return quantized indices
            commit_loss,          # Return commitment loss
            perplexity.detach(),  # For logging only
            active_num.detach(),  # For logging only
        )
