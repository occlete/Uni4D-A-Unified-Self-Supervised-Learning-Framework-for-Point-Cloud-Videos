import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import modules.pointnet2_utils as pointnet2_utils
from typing import List
def knn1(anchor_xyz:Tensor,xyz:Tensor,k:int)->Tensor:
    """
    Input:anchor_xyz:FPS down_samping (B,N/S,3)
          xyz:origional  (B,N,3)
          k:parmer
    Output:
          最近的K个索引
    """
    device = anchor_xyz.device
    B, N1, _ = anchor_xyz.shape
    N = xyz.shape[1]
    # b n/s 1 3 - b 1 n 3 广播机制 b,n/s n 64*2048 表示每个降采样的点与原始点之间的距离
    distance=torch.sum((anchor_xyz.unsqueeze(2)-xyz.unsqueeze(1))**2,dim=-1) #距离矩阵，shape为（b,n/s,n）
    _,idx=torch.topk(distance,k=k,largest=False,dim=-1)
    
    return idx #b n/s k

  
class Point_Transformer(nn.Module):  
    def __init__(self, 
                 in_planes: int,
                 spatial_kernel_size: [float, int],
                 mlp_planes: List[int],
                 ):          
        super(Point_Transformer, self).__init__()  
        self.r,self.k=spatial_kernel_size
        self.linear_q=nn.Linear(3,mlp_planes[0])#q 的映射
        self.linear_k=nn.Linear(3,mlp_planes[0])#k 的映射
        self.linear_v=nn.Linear(3,mlp_planes[0])#k 的映射
        self.dis_encode=nn.Linear(3,mlp_planes[0])
        self.mlp_attention_weight=nn.Sequential(
                                                nn.Linear(mlp_planes[0],mlp_planes[0]),
                                                )
        self.convflow=nn.Conv2d(in_channels=mlp_planes[0],out_channels=mlp_planes[0],kernel_size=1,bias=False)
        self.bn=nn.BatchNorm1d(mlp_planes[0])
    def forward(self, anchor_xyzs, neighbor):  
        npoint1=neighbor.size(1)
        batch_size,npoints,ndims = anchor_xyzs.size()   
        device = anchor_xyzs.get_device()  
        
        anchor_xyzQ=self.linear_q(anchor_xyzs)#b n/s mlp_planes[0]
        Q=anchor_xyzQ.view(batch_size,npoints,1,-1).repeat(1,1,self.k,1)

        
        
        neighbor_k=self.linear_k(neighbor)
       # neighbor_v=self.linear_v(neighbor)#b n k mlp_planes[0]
        neighbor_features_k=neighbor_k.transpose(1, 2).contiguous()
       # neighbor_features_v=neighbor_v.transpose(1, 2).contiguous()

        #temporal
        idx_before = pointnet2_utils.ball_query(self.r, self.k, neighbor, anchor_xyzs) #索引
        neighbor_flipped = neighbor.transpose(1, 2).contiguous()                                                    # (B, 3, N)
        neighbor_grouped = pointnet2_utils.grouping_operation(neighbor_flipped, idx_before) #b 3 n/s K
        neighbor_query_ball=neighbor_grouped.permute(0,2,3,1) #b n/s k 3



        features_K=pointnet2_utils.grouping_operation(neighbor_features_k , idx_before).permute(0,2,3,1).contiguous()
        #features_v=pointnet2_utils.grouping_operation(neighbor_features_v , idx_before).permute(0,2,3,1).contiguous()


        
        y = anchor_xyzs.view(batch_size, npoints, 1, ndims).repeat(1, 1, self.k, 1)  # (b, n/s, k, ndims)  
        dis=self.dis_encode(y-neighbor_query_ball)
        #flow_weight=Q*features_K
        #flow_embed=torch.cat((dis,flow_weight),dim=-1)
        #out=self.convflow(flow_embed.permute(0,3,1,2))
        #out=torch.max(input=out,dim=-1,keepdim=False)[0].permute(0,2,1)
        #dis=y-neighbor_query_ball
        #dis=self.convdis(dis.permute(0,3,1,2))#b n k d-> b d n k->b mlp n k

        #attention
        attention_weight=self.mlp_attention_weight(Q-features_K)
        
        attention_weight=F.softmax(attention_weight,dim=-1)
        
        #attention=torch.einsum('b n k d, b n k d->b n d',attention_weight,value)
        attention=torch.max(input=dis*attention_weight,dim=2,keepdim=False)[0]
        #attention=self.bn(attention.permute(0,2,1)).permute(0,2,1)
        
        return attention

class FlowEmbedding(nn.Module):  
    def __init__(self, 
                 in_planes: int,
                 spatial_kernel_size: [float, int],
                 mlp_planes: List[int],
                 ):          
        super(FlowEmbedding, self).__init__()  
        self.in_planes=in_planes
        self.r,self.k=spatial_kernel_size
        
        #self.convf=nn.Conv2d(in_channels=in_planes+3,out_channels=mlp_planes[0],kernel_size=1,bias=False)
        self.convf=nn.Sequential(nn.Conv2d(in_channels=in_planes+3,out_channels=mlp_planes[0],kernel_size=1,bias=False),nn.ReLU(inplace=True),nn.BatchNorm2d(num_features=mlp_planes[0]))
        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=False))
            #if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            #if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)
    def forward(self, anchor_xyzs, neighbor,neighbor_features,features):  
        start_time=time.time()
        batch_size,npoints,ndims = anchor_xyzs.size()   
        device = anchor_xyzs.get_device()  
        
        """
            input   
                neighbor:(batch_size,n/s1,3)
                anchor_xyzs:(batch_size,n/s2,3)
                neighbor_features:(batch_size,n/s1,dim)
                features:(batch_size,n/s1,dim)
        output:
                anchor_xyz:(batch_size,n/s2,3)
                feature_s:(batch_size,n/s2,outchannel)
                
        """
        #temporal
        #distance 坐标维
        idx = pointnet2_utils.ball_query(self.r, self.k, neighbor, anchor_xyzs) #索引
        neighbor_flipped = neighbor.transpose(1, 2).contiguous()                                                    # (B, 3, N)
        neighbor_grouped = pointnet2_utils.grouping_operation(neighbor_flipped, idx) #b 3 n/s K
        neighbor_query_ball=neighbor_grouped.permute(0,2,3,1) #b n/s k 3
        y = anchor_xyzs.view(batch_size, npoints, 1, ndims).repeat(1, 1, self.k, 1)
        xyz_dis=y-neighbor_query_ball
        #xyz_dis=torch.cat((xyz_dis,t_dis),dim=-1)
        #print(dis_encode.shape)
        #特征维：
        features=features.transpose(1, 2).contiguous()
        neighbor_features=neighbor_features.transpose(1, 2).contiguous()
        features_Q=pointnet2_utils.grouping_operation(features , idx).permute(0,2,3,1).contiguous()    #b dim n/s2 k ->b n/s2 k dim 
        features_KV=pointnet2_utils.grouping_operation(neighbor_features , idx).permute(0,2,3,1).contiguous()   #b dim n/s2 k ->b n/s2 k dim
        feature_dis=features_Q*features_KV
        feature_new=torch.cat((xyz_dis,feature_dis),dim=-1)#batchsize,npoints,inplanes+3
        out=self.convf(feature_new.permute(0,3,1,2))
        out=self.mlp(out)
        #print(out.shape)
        out=torch.max(input=out,dim=-1,keepdim=False)[0].permute(0,2,1)
        end_time=time.time()
        #print("scfe forward time",end_time-start_time)
        return out
        
        


