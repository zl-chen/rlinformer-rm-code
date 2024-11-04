import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # 计算QK attention score，并返回最有用的几个 q 的 index
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape 
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 把原来的K（B, H, L_K, E）增加一个L_Q维度
        # unsqueeze后是(B, H, 1, L_K, E) ，expand就是复制到(B, H, L_Q, L_K, E) 
        # 所有的(B, H, i, L_K, E)都相同，i 属于【0，L_Q）
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        
        # index_sample是在【0，L_K）这个范围内，随机采样出L_Q x sample_k个整数
        # 得到shape为(L_Q, sample_k)的采样矩阵
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        
        # torch.arange(L_Q)得到[0，L_Q)共L_Q个连续正整数，unsqueeze后，shape为(96,1)
        # K_sample的shape为[32, 8, 96, 25, 64]
        # 看来这个torch.arange(L_Q).unsqueeze(1)是必要的
        # 和index_sample搭配，起到的目的是为每个query都去选25个不一样的keys 
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        # Q.unsqueeze(-2).shape, torch.Size([32, 8, 96, 1, 64])
        # K_sample.transpose(-2, -1).shape, torch.Size([32, 8, 96, 64, 25])
        # Q_K_sample.shape, torch.Size([32, 8, 96, 25])       
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        # torch.arange(B)[:, None, None].shape -> torch.Size([32, 1, 1])
        # torch.arange(H)[None, :, None].shape -> torch.Size([1, 8, 1])
        # M_top.shape -> torch.Size([32, 8, 25])
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        # 做矩阵乘法
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # （batch size，seq_len, head_num, hidden）
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        #  AttentionLayer传进来的queries, keys, values的shape都是[32, 96, 8, 64]，
        # 也就是（batch size，seq_len, head_num, hidden），
        # 三个都经过transpose后变成，（batch size, head_num, seq_len, hidden），也即（32, 8, 96, 64）
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # U_part是c*ln(L_k)，u是c*ln(L_q),即选取的 query和 key的个数
        # c是一个超参数，此处取的是5，L_k是key的长度，L_q是query的长度
        # ln96=4.56，ceil(4.56)=5，所以U_part=5*5=25，u=25  (注意：np.log是以e为底的)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        # self._prob_QK用于计算QK对应的score，还有top k个query对应的index（其实这个index也对应着score非0的地方）
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


# 在里面调用probattention，之所以再加一层是为了方便做对比的时候，更换成其他的注意力方式
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        # 默认用的就是prob attention
        self.inner_attention = attention
        # 改变的是tensor的最后一维，将d_model映射成 d_keys*n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # 输出的时候，再映射回来
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 先调用上面的映射函数，然后再reshape成多头的形式
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
