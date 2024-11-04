import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# 2.1 位置编码（0-99），即encoder输入的长度，decoder同理
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("pe1",self.pe.shape)
        # print("pe1",self.pe)
        # print("pe1.1",self.pe[:, :x.size(1)].shape)
        # print("pe1.1",self.pe[:, :x.size(1)])
        
        return self.pe[:, :x.size(1)]
    
    
    

# 2.2 全局性质的位置编码（0-499）
class PositionalEmbeddingAll(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbeddingAll, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # 该组参数在模型训练时不会更新（即调用optimizer.step()后该组参数不会变化，只可人为地改变它们的值）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("ahh",x.shape)
        # print("ahh",x)
        start=int(x[0,0,0])
        # 感觉有问题啊，像是行列写反了一样
        # 实际没问题，因为上面 pe = pe.unsqueeze(0)，在最前面加了一个维度
        # print("pe2",self.pe.shape)
        # print("pe2",self.pe)
        return self.pe[:, start:start+x.size(1)]
    
    
    

# 1.数据embedding，将输入维度映射成 d_model维（原文中是7维映射到512维度）
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # print("wordOrigin",x.shape)
        # print("wordOrigin",x)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        # print("word32",x.shape)
        # print("word32",x)
        return x

# 属于全局时间embedding的一种，具体来说，是全局时间embedding中的TemporalEmbedding的一种
# 用作映射，将c_in映射成d_model（原模型为512）
class FixedEmbedding(nn.Module):
    # 此处c_in与下方c_in似乎不同，此处由TemporalEmbedding传入，如 minute_size
    def __init__(self, c_in , d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


# 3.时间戳编码 TemporalEmbedding 与 TimeFeatureEmbedding
# 前者使用month_embed、day_embed、weekday_embed、hour_embed和minute_embed(可选)多个embedding层处理输入的时间戳，将结果相加；
# 后者直接使用一个全连接层将输入的时间戳映射到512维的embedding。
# TemporalEmbedding中又分为两种，分别是FixedEmbedding以及pytorch自带的embedding，其中FixedEmbedding，它使用位置编码作为embedding的参数，不需要训练参数。
# 而后者，Pytorch自带的embedding层，再训练参数，
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class  DataEmbedding(nn.Module):
    # c_in可能是encoder的输入特征数，也可能是decoder的输入特征数，具体看外面传
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.position_embeddingAll = PositionalEmbeddingAll(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 三个embedding直接相加得到模型输入，分别为数据embedding，位置embedding以及时间embedding
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        # 暂时先删除时间embedding？？？
        
        # x = self.value_embedding(x) + self.position_embedding(x) + self.position_embeddingAll(x_mark)
        # 去除 1-100的位置embedding
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x)