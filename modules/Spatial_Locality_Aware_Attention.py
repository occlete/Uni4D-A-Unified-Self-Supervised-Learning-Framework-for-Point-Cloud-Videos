import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pointnet2_utils
import numpy as np

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,mask_ratio=0.75):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask_ratio=mask_ratio
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def random_masking(self, x):
        B, G, _,_ = x.shape

        if self.mask_ratio == 0:
            return torch.zeros(x.shape[:3]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(x.device)  # B G

    def forward(self, xyzs, feature):
        """
        xyzs (bl) n 3
        feature (bl) n d
        """
        b,n, _, h = *feature.shape, self.heads
        device = xyzs.get_device()
        norm_features = self.norm(feature)
        q,k,v = self.to_qkv(norm_features).chunk(3, dim = -1)


        idx = pointnet2_utils.ball_query(0.3,8, xyzs, xyzs)#c n k
            
        key_filp=k.transpose(1, 2).contiguous() #c d n
        #print(key_filp.shape)
        key_grouping=pointnet2_utils.grouping_operation(key_filp, idx)#c d n k
            
        values_filp=v.transpose(1, 2).contiguous()
        value_grouping=pointnet2_utils.grouping_operation(values_filp, idx)#c d n k
            
        neighbor_xyz_filp = xyzs.transpose(1, 2).contiguous()#c 3 n
        xyz_grouping=pointnet2_utils.grouping_operation(neighbor_xyz_filp, idx)#c 3 n k
        spatial_temporal_displacements=xyz_grouping.permute(0,2,3,1)-xyzs.unsqueeze(2)  #c n k 3
        displacements = rearrange(spatial_temporal_displacements, 'b n k d->(b n) k d', b=b,n=n, k=8)
        bn,k,coor=displacements.shape
            


        keys = rearrange(key_grouping.permute(0, 2, 3, 1), 'b n k (h d)->(b n) h k d', b=b,h=h, k=8)

        values = rearrange(value_grouping.permute(0,2,3,1), 'b n k (h d)->(b n) h k d',b=b,h=h, k=8)
        
        query=rearrange(q,'b n (h d) -> (b n) h 1 d',h=h)#b n d


        attn_weight=torch.einsum('c h i d,c h j d->c h i j',query,keys)*self.scale #(b l n) h 1 k
            
        attn = attn_weight.softmax(dim=-1)
            
            
        attn_dis=attn.unsqueeze(4)#(b l n) h 1 k 1
            
        dis=displacements.unsqueeze(1).unsqueeze(2)#(b l n) 1 1 k d 
        dis_attn=attn_dis*dis#c h i j d
            
        dis_attn=torch.max(input=dis_attn,dim=2,keepdim=False)[0]
            
        dis_attn=torch.max(input=dis_attn,dim=2,keepdim=False)[0]#c h 3
        dis_attn=self.spatial_op(dis_attn)
        dis_attn=rearrange(dis_attn,'(b n) h d->b n (h d)',n=n,h=h)
            
        attnout=torch.einsum('c h i j,c h j d->c h i d',attn,values)
        attnout=rearrange(attnout,'(b n) h 1 d->b n (h 1 d)',n=n,h=h)
        attnout=attnout+dis_attn
            
        out =  self.to_out(attnout)#b 1 d
        
        return out+feature

class SLAA(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features
        
if __name__ == '__main__':
    # Create random input tensors
    x = torch.zeros(2*12, 32, 3).cuda()  # batch size of 2, 96 tokens, dimension 64
    y = torch.zeros(2*12, 32, 1024).cuda()  # batch size of 2, 288 tokens, dimension 64

    # Initialize the attention layers with correct dimensions
    attention_model = SLAA().cuda()

    # Test self-attention
    attention_output = attention_model(x,y)  # self-attention
    print("Self-attention output shape:", attention_output.shape)

    