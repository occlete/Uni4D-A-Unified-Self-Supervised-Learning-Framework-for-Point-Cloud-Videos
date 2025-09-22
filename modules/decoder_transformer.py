import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        b, n, _, h = *x.shape, self.heads

        norm_features_q = self.norm(x)
    
        #b, n, _, h = *x.shape, self.heads
        _, m, _,=y.shape
        norm_features_kv=self.norm(y)
        qkv = self.to_qkv(norm_features_q).chunk(3, dim = -1)
        qkv2=self.to_qkv2(norm_features_kv).chunk(3, dim = -1)
        q_v, k_v, v_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q_m,k_m,v_m=map(lambda t: rearrange(t, 'b m (h d) -> b h m d', h = h), qkv2)
        #k=torch.cat((k_v,k_m),dim=2)
        #v=torch.cat((v_v,v_m),dim=2)
        dots_r = einsum('b h i d, b h j d -> b h i j', q_v, k_m) * self.scale
        dots_f = einsum('b h i d, b h j d -> b h i j', q_m, k_v) * self.scale
        
        attn_r = dots_r.softmax(dim=-1)
        attn_f = dots_f.softmax(dim=-1)

        out_r = einsum('b h i j, b h j d -> b h i d', attn_r, v_m)
        out_f = einsum('b h i j, b h j d -> b h i d', attn_f, v_v)
        out_r = rearrange(out_r, 'b h n d -> b n (h d)')
        out_f = rearrange(out_f, 'b h n d -> b n (h d)')
        out_r =  self.to_out(out_r)
        out_f =  self.to_out(out_f)
        return out_r+x,out_f+y
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x,y):
        b,n,c=x.shape
        for attn,crossatten, ff in self.layers:
            z = torch.cat((x,y),dim=0)
            z = attn(z)
            x,y=z[0:b,:,:],z[b:,:,:]
            x,y = crossatten(x,y)
            x,y = ff(x),ff(y)
        return x,y
        

if __name__ == '__main__':
    # Create random input tensors
    x = torch.zeros(2, 96, 64).cuda()  # batch size of 2, 96 tokens, dimension 64
    y = torch.zeros(2, 288, 64).cuda()  # batch size of 2, 288 tokens, dimension 64

    # Initialize the attention layers with correct dimensions
    attention_model = Attention(dim=64).cuda()

    # Test self-attention
    attention_output = attention_model(x)  # self-attention
    print("Self-attention output shape:", attention_output.shape)

    # Test cross-attention
    cross_attention_output = attention_model(x, y)  # cross-attention
    print("Cross-attention output shape:", cross_attention_output.shape)
