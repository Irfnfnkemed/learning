import torch
import triton
from kernel import _attn_fwd, _attn_bwd_preprocess, _attn_bwd_dkdv, _attn_bwd_dq

class _attention(torch.autograd.Function):

    @staticmethod
    # q,k,v: [B,H,N,D]
    def forward(ctx, q, k, v, causal):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            BATCH=q.shape[0], HEAD_NUM=q.shape[1], N_CTX=q.shape[2], HEAD_DIM=HEAD_DIM_K,
            STAGE=stage)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        PRE_BLOCK = 128
        STAGE = 3 if ctx.causal else 1
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta, 
            BATCH, N_HEAD, N_CTX, 
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd_dkdv[grid](
            q, k, v, do, dk, dv, M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, N_CTX, HEAD_DIM=ctx.HEAD_DIM, 
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1,
            STAGE=STAGE
        )
        grid = (N_CTX // BLOCK_M2, 1, BATCH * N_HEAD)
        _attn_bwd_dq[grid](
            q, k, v, do, dq, M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, N_CTX, HEAD_DIM=ctx.HEAD_DIM, 
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
            STAGE=STAGE
        )
        return dq, dk, dv, None, None