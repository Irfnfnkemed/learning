import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, M, Out,  #
              stride_b, stride_h, stride_n, stride_d, 
              BATCH, HEAD_NUM, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, 
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr 
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM) # ?
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // HEAD_NUM
    off_h = off_bh % HEAD_NUM
    qvk_offset = off_b.to(tl.int64) * stride_b + off_h.to(tl.int64) * stride_h

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_d, stride_n),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # initialize offsets
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 
                                        4 - STAGE, off_m, off_n, N_CTX)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, off_m, off_n, N_CTX)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_bh * N_CTX + off_m
    
    # write back
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, off_m: tl.constexpr, off_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    log2e = 1.44269504 
    # causal = True, range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, tl.maximum(start_m * BLOCK_M - BLOCK_N, 0)
    elif STAGE == 2:
        lo, hi = tl.maximum(start_m * BLOCK_M - BLOCK_N, 0), (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_N)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    if STAGE == 2:
        for start_n in range(lo, hi, BLOCK_N):
            # Compute qk
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k)
            mask = off_m[:, None] >= (start_n + off_n[None, :])
            qk = qk * log2e + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # Update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # Update output accumulator
            acc = acc * alpha[:, None]
            # Update acc
            v = tl.load(V_block_ptr)
            acc = tl.dot(p.to(tl.float16), v, acc)
            # Update m_i and l_i
            m_i = m_ij
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    else:
        for start_n in range(lo, hi, BLOCK_N):
            # Compute qk
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * log2e)
            qk = qk * log2e - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # Update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # Update output accumulator
            acc = acc * alpha[:, None]
            # Update acc
            v = tl.load(V_block_ptr)
            acc = tl.dot(p.to(tl.float16), v, acc)
            # Update m_i and l_i
            m_i = m_ij
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N)) 
    return acc, l_i, m_i

@triton.jit
def _attn_bwd_dkdv(Q, K, V, DO, DK, DV, M, D,
              stride_b, stride_h, stride_n, stride_d,
              HEAD_NUM, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
              STAGE
              ):

    # initialize offsets
    off_bh = tl.program_id(2)
    off_b = off_bh // HEAD_NUM
    off_h = off_bh % HEAD_NUM
    qvk_offset = off_b.to(tl.int64) * stride_b + off_h.to(tl.int64) * stride_h
    m_offset = qvk_offset // HEAD_DIM
    start_n = tl.program_id(0)   
    
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(N_CTX,),
        strides=(stride_n // HEAD_DIM,),
        offsets=(0,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D + m_offset,
        shape=(N_CTX,),
        strides=(stride_n // HEAD_DIM,),
        offsets=(0,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )

    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    
    off_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    off_m = tl.arange(0, BLOCK_M)

    if STAGE & 1:
        dk, dv = _attn_bwd_dkdv_inner(dk, dv, Q_block_ptr, k, v,
                                DO_block_ptr, M_block_ptr, D_block_ptr, 
                                stride_n, stride_d, 
                                HEAD_NUM, N_CTX, HEAD_DIM, 
                                BLOCK_M, BLOCK_N, 
                                start_n, off_m, off_n, 
                                STAGE = 4 - STAGE 
                                )
    if STAGE & 2:
        dk, dv = _attn_bwd_dkdv_inner(dk, dv, Q_block_ptr, k, v,
                                DO_block_ptr, M_block_ptr, D_block_ptr, 
                                stride_n, stride_d, 
                                HEAD_NUM, N_CTX, HEAD_DIM, 
                                BLOCK_M, BLOCK_N, 
                                start_n, off_m, off_n, 
                                STAGE = 2 
                                )
        
    # write back
    tl.store(DV_block_ptr, dv.to(tl.float16))
    tl.store(DK_block_ptr, dk.to(tl.float16))
    
@triton.jit
def _attn_bwd_dq(Q, K, V, DO, DQ, M, D,
              stride_b, stride_h, stride_n, stride_d,
              HEAD_NUM, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
              STAGE
              ):
    
    # initialize offsets
    off_bh = tl.program_id(2)
    off_b = off_bh // HEAD_NUM
    off_h = off_bh % HEAD_NUM
    qvk_offset = off_b.to(tl.int64) * stride_b + off_h.to(tl.int64) * stride_h
    m_offset = qvk_offset // HEAD_DIM
    start_m = tl.program_id(0)
    
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_n, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(N_CTX,),
        strides=(stride_n // HEAD_DIM,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D + m_offset,
        shape=(N_CTX,),
        strides=(stride_n // HEAD_DIM,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load Q,m,delta,do: they stay in SRAM throughout the inner loop.
    q = tl.load(Q_block_ptr)
    m = tl.load(M_block_ptr)
    delta = tl.load(D_block_ptr)
    do = tl.load(DO_block_ptr)
    
    off_n = tl.arange(0, BLOCK_N) 
    off_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M

    if STAGE & 1:
        dq = _attn_bwd_dq_inner(dq, q, K_block_ptr, V_block_ptr, do, m, delta, 
                                stride_n, stride_d, 
                                HEAD_NUM, N_CTX, HEAD_DIM, 
                                BLOCK_M, BLOCK_N, 
                                start_m, off_m, off_n, 
                                STAGE=4-STAGE 
                                )
    if STAGE & 2:
        dq = _attn_bwd_dq_inner(dq, q, K_block_ptr, V_block_ptr, do, m, delta, 
                                stride_n, stride_d, 
                                HEAD_NUM, N_CTX, HEAD_DIM,
                                BLOCK_M, BLOCK_N,  
                                start_m, off_m, off_n, 
                                STAGE=2 
                                )
        
    # write back
    tl.store(DQ_block_ptr, dq.to(tl.float16))
    
    
# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv_inner(dk, dv, Q_block_ptr, k, v, DO_block_ptr, 
                   M_block_ptr, D_block_ptr,
                   stride_n, stride_d, 
                   HEAD_NUM, N_CTX, HEAD_DIM: tl.constexpr, 
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   start_n, off_m, off_n,
                   STAGE: tl.constexpr):
    
    log2e = 1.44269504 
    # causal = True, range of values handled by this stage
    if STAGE == 2:
        lo, hi = tl.minimum(((start_n + 1) * BLOCK_N) // BLOCK_M * BLOCK_M, N_CTX), N_CTX
    elif STAGE == 1:
        lo, hi = (start_n * BLOCK_N) // BLOCK_M * BLOCK_M, tl.minimum((start_n + 1) * BLOCK_N, N_CTX)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    M_block_ptr = tl.advance(M_block_ptr, (lo,))
    D_block_ptr = tl.advance(D_block_ptr, (lo,))
    if STAGE == 1:
        for start_m in range(lo, hi, BLOCK_M):
            # Compute P=softmax(QK^t) 
            q = tl.load(Q_block_ptr)
            qkT = tl.dot(k, tl.trans(q))
            m = tl.load(M_block_ptr)
            pT = tl.math.exp2(qkT * log2e - m[None, :])    
            mask = (start_m + off_m)[None, :] >= off_n[:, None]
            pT = tl.where(mask, pT, 0.0)
            # Compute dV.
            do = tl.load(DO_block_ptr)
            dv += tl.dot(pT.to(tl.float16), do)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do).to(tl.float16))
            delta = tl.load(D_block_ptr)
            dsT = pT * (dpT - delta[None, :])
            # Compute dK
            dk += tl.dot(dsT.to(tl.float16), q)
            Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
            DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
            M_block_ptr = tl.advance(M_block_ptr, (BLOCK_M,))
            D_block_ptr = tl.advance(D_block_ptr, (BLOCK_M,))
    else:
        for start_m in range(lo, hi, BLOCK_M):
            # Compute P=softmax(QK^t) 
            q = tl.load(Q_block_ptr)
            qkT = tl.dot(k, tl.trans(q))
            m = tl.load(M_block_ptr)
            pT = tl.math.exp2(qkT * log2e - m[None, :])    
            # Compute dV.
            do = tl.load(DO_block_ptr)
            dv += tl.dot(pT.to(tl.float16), do)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do).to(tl.float16))
            delta = tl.load(D_block_ptr)
            dsT = pT * (dpT - delta[None, :])
            # Compute dK
            dk += tl.dot(dsT.to(tl.float16), q)
            Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
            DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
            M_block_ptr = tl.advance(M_block_ptr, (BLOCK_M,))
            D_block_ptr = tl.advance(D_block_ptr, (BLOCK_M,))
            
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq_inner(dq, q, K_block_ptr, V_block_ptr, do, 
                   m, delta,
                   stride_n, stride_d, 
                   HEAD_NUM, N_CTX, HEAD_DIM: tl.constexpr, 
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   start_m, off_m, off_n,
                   STAGE):
    log2e = 1.44269504 
    # causal = True, range of values handled by this stage
    if STAGE == 2:
        lo, hi = 0, tl.maximum(start_m * BLOCK_M - BLOCK_N, 0)
    elif STAGE == 1:
        lo, hi = tl.maximum(start_m * BLOCK_M - BLOCK_N, 0), (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_N)
    else:
        lo, hi = 0, N_CTX
    
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    if STAGE == 1:
        for start_n in range(lo, hi, BLOCK_N):
            # Compute P=softmax(QK^t) 
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, tl.trans(k))
            p = tl.math.exp2(qk * log2e - m[:, None])    
            mask = off_m[:, None] >= (off_n + start_n)[None, :]
            p = tl.where(mask, p, 0.0)
            # Compute dP and dS.
            v = tl.load(V_block_ptr)
            dp = tl.dot(do, tl.trans(v).to(tl.float16))
            ds = p * (dp - delta[:, None])
            # Compute dQ
            dq += tl.dot(ds.to(tl.float16), k)
            K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    else:
        for start_n in range(lo, hi, BLOCK_N):
            # Compute P=softmax(QK^t) 
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, tl.trans(k))
            p = tl.math.exp2(qk * log2e - m[:, None])    
            # Compute dP and dS.
            v = tl.load(V_block_ptr)
            dp = tl.dot(do, tl.trans(v).to(tl.float16))
            ds = p * (dp - delta[:, None])
            # Compute dQ
            dq += tl.dot(ds.to(tl.float16), k)
            K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return dq


# compute O * dO
@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,
                         BATCH, HEAD_NUM, N_CTX,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_bh = tl.program_id(1)
    off_d = tl.arange(0, HEAD_DIM)
    off_o = off_bh * N_CTX * HEAD_DIM+ off_m[:, None] * HEAD_DIM + off_d[None, :]
    off_delta = off_bh * N_CTX + off_m
    # load
    o = tl.load(O + off_o)
    do = tl.load(DO + off_o).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_delta, delta)

