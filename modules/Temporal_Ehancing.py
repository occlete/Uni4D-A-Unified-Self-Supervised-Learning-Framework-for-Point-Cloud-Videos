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
from typing import List
from modules.Temporal_point_transformer import *
from modules.transformer import *
class Temporal(nn.Module):  
    def __init__(self, spatial_stride,emb_relu,
                 in_planes: int,
                 temporal_stride,
                 mlp_planes: List[int],
                 spatial_kernel_size: [float, int],
                 temporal_padding: [int, int] = [0, 0], 
                 temporal_padding_mode: str = 'replicate',):          
        super(Temporal, self).__init__()  
        self.in_planes = in_planes
        self.temporal_stride=temporal_stride
        self.spatial_stride=spatial_stride
        self.temporal_padding_mode=temporal_padding_mode
        self.temporal_padding = temporal_padding
        self.emb_relu = nn.ReLU() if emb_relu else False
        self.r, self.k = spatial_kernel_size
        self.em=nn.ReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=mlp_planes[0], kernel_size=1, bias=False))
        #self.kgcnlayer_nofeature=K_GCN(mlp_planes[0],[self.r,self.k])
        if self.in_planes!=0:
            #self.kgcnlayer_with_feature=K_GCNLayer(in_planes,mlp_planes[0],[self.r,self.k])
            self.temporal_point_transformer_with_feature=FlowEmbedding(in_planes,[self.r,self.k],mlp_planes)
        self.convf2=nn.Conv2d(in_channels=2*mlp_planes[0],out_channels=3,kernel_size=1,bias=False)
        self.transformer = Transformer(mlp_planes[0], 2, 4, 64, mlp_planes[0])
    def forward(self, xyzs, features=None):  
        start_time=time.time()
        batch_size = xyzs.size(0)  
        nframes = xyzs.size(1)   
        npoint1 = xyzs.size(2) 
        device = xyzs.get_device()  
        
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)  # 根据每个切片中的帧数进行分割  
        xyzs= [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]  
        #xyzs=[]
        #for t,xyz in enumerate(xyzs1):
        #    t=torch.ones(batch_size,npoint1,1, dtype=torch.float32, device=device)*(t+1)
        #    xyz = torch.cat(tensors=(xyz, t), dim=2)
        #    xyzs.append(xyz)
        #print(xyzs[0].shape)
        #print(xyzs[0])
        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]


        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]


        
        new_xyz=[]
        new_feature_temporal=[]
        #for i in range(1,nframes+1):  
        for i in range(1,len(xyzs)-1,self.temporal_stride): 
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[i], npoint1//self.spatial_stride) #FPS降采样  # (B, N//self.spatial_stride)
            anchor_xyz = pointnet2_utils.gather_operation(xyzs[i].transpose(1, 2).contiguous(), anchor_idx).transpose(1, 2).contiguous() # (B, N//spatial_stride, 3)
            #print(anchor_xyz.shape)
            new_flow=[]
            _, npoints, ndims = anchor_xyz.size()
            
            if self.in_planes!=0:
                """
                   with feature
                """
                #t_displacement_b = torch.ones((anchor_xyz.size()[0],  anchor_xyz.size()[1], self.k,1), dtype=torch.float32, device=device) * (-1)
                #attention_before=self.temporal_point_transformer_with_feature(anchor_xyz,xyzs[i-1],features[i-1],features[i])
                #new_flow.append(attention_before)
                #if i==len(xyzs)-2:
                #    t_displacement_a = torch.ones((anchor_xyz.size()[0],  anchor_xyz.size()[1], self.k,1), dtype=torch.float32, device=device) * (0)
                #else:
                #    t_displacement_a = torch.ones((anchor_xyz.size()[0],  anchor_xyz.size()[1], self.k,1), dtype=torch.float32, device=device) * (2)
                attention_after=self.temporal_point_transformer_with_feature(anchor_xyz,xyzs[i+1],features[i+1],features[i])
                #new_flow.append(attention_after)
                #new_temporal=torch.stack(tensors=new_flow,dim=1)
                #new_temporal=torch.sum(input=new_temporal,dim=1,keepdim=False)
                new_temporal=attention_after#+attention_before
                #new_temporal=self.convf2(new_features_temporal.permute(0,3,1,2)).permute(0,2,3,1)
                #attention_after+attention_before
                new_feature_temporal.append(new_temporal)

            else:
                attention_before=self.temporal_point_transformer(anchor_xyz,xyzs[i-1])
                #kneighborgcn_before=self.tkgcn(anchor_xyz,xyzs[i-1])
                #new_flow.append(attention_before)
                attention_after=self.temporal_point_transformer(anchor_xyz,xyzs[i+1])
                #kneighborgcn_after=self.tkgcn(anchor_xyz,xyzs[i+1])
                #new_flow.append(attention_after)
                #new_temporal=torch.stack(tensors=new_flow,dim=1)
                #new_temporal=torch.sum(input=new_temporal,dim=1,keepdim=False)
                #new_temporal=kneighborgcn_after+kneighborgcn_before
                new_temporal=attention_after+attention_before#b n/s k 3
                #new_temporal=self.conv1(new_temporal.permute(0,3,1,2))
                #new_temporal=torch.max(input=new_temporal,dim=-1,keepdim=False)[0].permute(0,2,1)
                new_feature_temporal.append(new_temporal)

            new_xyz.append(anchor_xyz) #b nframes n/s 3
        new_xyzs=torch.stack(tensors=new_xyz,dim=1) #b nframe n/s 3
        
        new_features_temporal=torch.stack(tensors=new_feature_temporal,dim=1) #b nframe,n c
        
        new_features=new_features_temporal
        end_time=time.time()
        #print("forward time",end_time-start_time)
        return new_xyzs,new_features
        
        
        


 