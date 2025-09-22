import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from modules.point_4d_convolution import *
from modules.SHOT import *
from modules.transformer import *
#from modules.transformerv1 import *
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL2
#from models.TextEncoder import TextEncoder
#from modules.Temporal_Enhancing import *
#from modules.decoder_transformer import *
#from models.model_utils import *


class ContrastiveMaskedAutoEncoder(nn.Module):
    def __init__(self, radius=0.1, nsamples=32, spatial_stride=32,                            # P4DConv: spatial
                 temporal_kernel_size=3, temporal_stride=3,                                   # P4DConv: temporal
                 en_emb_dim=1024, en_depth=10, en_heads=8, en_head_dim=256, en_mlp_dim=2048,  # encoder
                 de_emb_dim=512,  de_depth=5,  de_heads=8, de_head_dim=256, de_mlp_dim=1024,  # decoder 
                 mask_ratio=0.6,
                 num_classes=60,
                 dropout1=0.05,
                 dropout_cls=0.5,
                 temperature = 0.1,
                 momentum = 0.999,
                 pretraining=True,
                 vis=False,
                 ):
        super(ContrastiveMaskedAutoEncoder, self).__init__()

        self.pretraining = pretraining

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[en_emb_dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        #self.motion = MotionEmbedding(spatial_kernel_size=[radius, nsamples])
        # encoder        
        self.encoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=en_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu1 = nn.ReLU()
        
        self.emb_relu2 = nn.ReLU()
        self.encoder_transformer = Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
        self.encoder_norm = nn.LayerNorm(en_emb_dim)
        
        self.encoder_transformer2 = Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
        self.encoder_norm2 = nn.LayerNorm(en_emb_dim)
        
        
        self.vis = vis
        self.nsamples = nsamples
        self.tk = temporal_kernel_size
        self.shotL = getDescriptorLength(elevation_divisions=2,
                                        azimuth_divisions=4
        )
        if self.pretraining:
            
            self.global_transformer=Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
            self.global_norm=nn.LayerNorm(en_emb_dim)
            
            self.global_transformer2=Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
            self.global_norm2=nn.LayerNorm(en_emb_dim)
            # Mask
            self.mask_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
            trunc_normal_(self.mask_token, std=.02)
            self.feature_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
            trunc_normal_(self.feature_token, std=.02)
            

            # decoder
            self.decoder_embed = nn.Linear(en_emb_dim, de_emb_dim, bias=True)
            self.decoder_cl = nn.Linear(de_emb_dim, en_emb_dim, bias=True)
            self.decoder_frame = nn.Sequential(
                nn.Linear(en_emb_dim, 4*en_emb_dim, bias=True),
                nn.BatchNorm1d(4*en_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(4*en_emb_dim, en_emb_dim, bias=True),
                )#
            self.decoder_frame_backward = nn.Sequential(
                nn.Linear(en_emb_dim, 4*en_emb_dim, bias=True),
                nn.BatchNorm1d(4*en_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(4*en_emb_dim, en_emb_dim, bias=True),
                )#
            self.decoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=de_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)
            
            self.decoder_transformer = Transformer(de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1)

            self.decoder_norm = nn.LayerNorm(de_emb_dim)

            # points_predictor
            self.points_predictor = nn.Conv1d(de_emb_dim, 3 * nsamples * temporal_kernel_size, 1)
            self.decoder_feature=nn.Conv1d(de_emb_dim,en_emb_dim,1)

            # loss
            self.criterion_dist = ChamferDistanceL2().cuda()
            self.criterion_shot = torch.nn.SmoothL1Loss().cuda()
            self.criterion_cl = nn.CrossEntropyLoss().cuda()
            self.criterion_frame = nn.CrossEntropyLoss().cuda()
            
            self.mask_ratio = mask_ratio
            self.temperature=temperature
            self.momentum=momentum

        else:

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(en_emb_dim),
                nn.Linear(en_emb_dim, en_mlp_dim),
                nn.GELU(),
                nn.Linear(en_mlp_dim, num_classes),
             )
             
            
    def momentum_update(self):
        for param_b, param_m in zip(self.encoder_transformer.parameters(),
                                    self.global_transformer.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)
        
        for param_b, param_m in zip(self.encoder_transformer2.parameters(),
                                    self.global_transformer2.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)
        
    def random_masking(self, x):
        B, G, _ = x.shape
        
        if self.mask_ratio == 0:
            return torch.zeros(x.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        #visible_idx = []
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
            
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)


        return overall_mask.to(x.device)# B G

    
    def forward_encoder(self, clips,clips_aug=None):
        #print(clips.shape)
        # [B, L, N, 3]
        xyzs, features, xyzs_neighbors, shot_descriptors = self.tube_embedding(clips)
        #print(xyzs.shape)
        #print(features.shape)
        features = features.permute(0, 1, 3, 2)                                              # [B, L, N, C]    
        batch_size, L, N, C = features.shape
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]                # L*[B, N, 3]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((batch_size, N, 1), dtype=torch.float32, device=clips.device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)                                            # [B, L, N, 4]

        if self.pretraining:
            # Targets
            xyzs_neighbors = torch.reshape(input=xyzs_neighbors, shape=(batch_size, L*N, self.tk, self.nsamples, 3)) # [B, L*N, tk, nn, 3]
            shot_descriptors = torch.reshape(input=shot_descriptors, shape=(batch_size, L, N, self.tk, self.shotL))   # [B, L*N, tk, shotL]
            
            # Masking
            xyzts=xyzts.reshape(batch_size*L,N,4)
            bool_masked_pos = self.random_masking(xyzts)
            bool_masked_pos = bool_masked_pos.reshape(batch_size,-1)
            xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 7]
            features = torch.reshape(input=features, shape=(batch_size, L*N, C))
            # Encoding the visible part
            pos_emb_full = self.encoder_pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
            
            
            fea_emb_vis = features[~bool_masked_pos].reshape(batch_size*L*8, -1, C)
            pos_emb_vis = pos_emb_full[~bool_masked_pos].reshape(batch_size*L*8, -1, C)
            features = features.reshape(batch_size*L*8, -1, C)
            pos_emb_full = pos_emb_full.reshape(batch_size*L*8, -1, C)
            
            fea_emb_vis = fea_emb_vis + pos_emb_vis
            fea_emb_full = features + pos_emb_full
            """
            first transformer for reconstruction
            """
            fea_emb_vis = self.emb_relu1(fea_emb_vis)
            fea_emb_vis = self.encoder_transformer(fea_emb_vis)
            #fea_emb_vis = self.encoder_norm(fea_emb_vis)#(batch_size*L*8, -1, C)
            
            """
            second transformer for contrastive learning
            """
            primitive_feature_vis = fea_emb_vis.permute(0, 2, 1)
            primitive_feature_vis = F.adaptive_max_pool1d(primitive_feature_vis, (1))  # B*l*4, C, 1
            primitive_feature_vis = torch.reshape(input=primitive_feature_vis, shape=(batch_size, L * 8, C))  # [B, L*4, C]
            anchor_feature_vis = torch.reshape(input=primitive_feature_vis, shape=(batch_size*L, 8, C))
            anchor_feature_vis = anchor_feature_vis.permute(0, 2, 1)
            anchor_feature_vis = F.adaptive_max_pool1d(anchor_feature_vis, (1))
            anchor_feature_vis = torch.reshape(input=anchor_feature_vis, shape=(batch_size, L, C))

            primitive_feature_vis = self.emb_relu2(anchor_feature_vis)
            fea_emb_vis_transformer = self.encoder_transformer2(primitive_feature_vis)#b l c
            fea_emb_vis_transformer = self.encoder_norm2(primitive_feature_vis)#b l c
            
            """
            first transformer for reconstruction
            """
            fea_emb_full = self.emb_relu1(fea_emb_full)
            fea_emb_full = self.global_transformer(fea_emb_full)
            #fea_emb_full = self.global_norm(fea_emb_full)
            
            """
            second transformer for contrastive learning
            """
            primitive_feature_full = fea_emb_full.permute(0, 2, 1)
            primitive_feature_full = F.adaptive_max_pool1d(primitive_feature_full, (1))  # B*l*4, C, 1
            primitive_feature_full = torch.reshape(input=primitive_feature_full, shape=(batch_size, L * 8, C))  # [B, L*4, C]
            anchor_feature_full = torch.reshape(input=primitive_feature_full, shape=(batch_size*L, 8, C))
            anchor_feature_full = anchor_feature_full.permute(0, 2, 1)
            anchor_feature_full = F.adaptive_max_pool1d(anchor_feature_full, (1))
            anchor_feature_full = torch.reshape(input=anchor_feature_full, shape=(batch_size, L, C))

            primitive_feature_full = self.emb_relu2(anchor_feature_full)
            fea_emb_full_transformer = self.global_transformer2(primitive_feature_full)#b l c
            fea_emb_full_transformer = self.global_norm2(primitive_feature_full)#b l c
            
            
            return fea_emb_vis, fea_emb_vis_transformer,bool_masked_pos, xyzts, xyzs_neighbors,fea_emb_full,fea_emb_full_transformer,L

        else:
            # Token sequence
            xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 4]
            features = torch.reshape(input=features, shape=(batch_size, L*N, C))
            xyzts = self.encoder_pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

            embedding = xyzts + features

            point_feat = torch.reshape(input=embedding, shape=(batch_size * L * 8, -1, C))  # [B*L*4, n', C]
            point_feat = self.emb_relu1(point_feat)
            point_feat = self.encoder_transformer(point_feat)  # [B*L*4, n', C]
            #point_feat = self.encoder_norm(point_feat)
            primitive_feature = point_feat.permute(0, 2, 1)
            primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
            primitive_feature = torch.reshape(input=primitive_feature, shape=(batch_size, L * 8, C))  # [B, L*4, C]

            anchor_feature = torch.reshape(input=primitive_feature, shape=(batch_size*L, 8, C))
            anchor_feature = anchor_feature.permute(0, 2, 1)
            anchor_feature = F.adaptive_max_pool1d(anchor_feature, (1))
            anchor_feature = torch.reshape(input=anchor_feature, shape=(batch_size, L, C))

            primitive_feature = self.emb_relu2(anchor_feature)


            primitive_feature = self.encoder_transformer2(primitive_feature) # B. L*4, C
            primitive_feature = self.encoder_norm2(primitive_feature) # B. L*4, C
            #primitive_feature = primitive_feature.reshape(batch_size*L,-1)  

            output =  self.mlp_head(primitive_feature)
            #output = output.reshape(batch_size,L,-1)  # 增加 channel 维度

            return output              


    def forward_decoder(self, emb_vis,emb_vis_transformer, mask, xyzts, xyzs_neighbors,fea_emb_full,emb_full_transformer,L):
        device = xyzts.get_device()
        

        emb_vis = self.decoder_embed(emb_vis)#(batch_size*L*8, -1, C)
        #batch
        batch_Length_Primitive, N_vis, C_decoder = emb_vis.shape#b N_vis 512
        batch_size, Length,C = emb_full_transformer.shape#b 150 1024
        emb_full_transformer = emb_full_transformer.detach()
        #print(fea_emb_full.shape)
        pos_emd_vis = xyzts[~mask].reshape(batch_Length_Primitive, -1, 4)#batch_Length_Primitive 1
        pos_emd_mask = xyzts[mask].reshape(batch_Length_Primitive, -1, 4)#batch_Length_Primitive 3
        pos_emd_vis = self.decoder_pos_embed(pos_emd_vis.permute(0, 2, 1)).permute(0, 2, 1)
        pos_emd_mask = self.decoder_pos_embed(pos_emd_mask.permute(0, 2, 1)).permute(0, 2, 1)#b l*masked D
        
        _,N_masked,_ = pos_emd_mask.shape


        # 更新队列

        loss_predict = 0

        for b in range(batch_size):
            source_feature = emb_vis_transformer[b]#l 1024

            target_feature = emb_full_transformer[b]#l 1024

            predict_feature = self.decoder_frame(source_feature[:-1])#l-1 512 -> l-1 1024

            predict_feature_new = predict_feature
            key_feature_new = target_feature[1:]#l-1 1024

            predict_feature = torch.nn.functional.normalize(predict_feature_new, dim=1)
            key_feature = torch.nn.functional.normalize(key_feature_new, dim=1)

            logits = torch.mm(predict_feature, key_feature.transpose(1, 0))#[l-1,l-1]

            labels = torch.arange(key_feature.size()[0])

            labels = labels.cuda()
            loss_tmp = self.criterion_frame(logits / self.temperature, labels)

            loss_predict = loss_tmp + loss_predict

        loss_predict = loss_predict / batch_size

        loss_predict_backward = 0
        for b in range(batch_size):
            source_feature = emb_vis_transformer[b]#l 512
            target_feature = emb_full_transformer[b]#l 1024

            predict_feature = self.decoder_frame_backward(source_feature[1:])#l-1 512 -> l-1 1024

            predict_feature_new = predict_feature
            key_feature_new = target_feature[:-1]#l-1 1024

            predict_feature = torch.nn.functional.normalize(predict_feature_new, dim=1)
            key_feature = torch.nn.functional.normalize(key_feature_new, dim=1)

            logits = torch.mm(predict_feature, key_feature.transpose(1, 0))#[l-1,l-1]
            
            labels = torch.arange(key_feature.size()[0])
            labels = labels.cuda()
            loss_tmp = self.criterion_frame(logits / self.temperature, labels)

            loss_predict_backward = loss_tmp + loss_predict_backward

        loss_predict_backward = loss_predict_backward / batch_size
        
        loss_predict = (loss_predict+loss_predict_backward)*0.5
        # append masked tokens to sequence
        mask_tokens = self.mask_token.expand(batch_Length_Primitive, N_masked, -1)
        feature_tokens = self.feature_token.expand(batch_Length_Primitive, N_masked, -1)

        emb_all = torch.cat([emb_vis, mask_tokens], dim=1)
        emb_all_feature = torch.cat([emb_vis, feature_tokens], dim=1)
        
        pos_all = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        emb_all = emb_all + pos_all
        emb_all_feature = emb_all_feature + pos_all

        emb_all = torch.cat((emb_all,emb_all_feature),dim=0)
        #emb_all = emb_all_feature
        emb_all= self.decoder_transformer(emb_all)  
        emb_all = self.decoder_norm(emb_all)
        
        emb_points,emb_features = emb_all[0:batch_Length_Primitive,:,:],emb_all[batch_Length_Primitive:,:,:]
        
        

        masked_emb = emb_points[:, -N_masked:, :]       # [B, M, C]

        masked_emb = masked_emb.transpose(1, 2)      # [B, C, M]
        
        # reconstruct points
        pre_points = self.points_predictor(masked_emb).transpose(1, 2)   
        #print(pre_points.shape)
        
        pre_points = pre_points.reshape(batch_Length_Primitive*N_masked, self.tk, self.nsamples, 3)                     # [B*M, tk, nn, 3]
        pred_list = torch.split(tensor=pre_points, split_size_or_sections=1, dim=1)     
        pred_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in pred_list] 
        
        # forward Loss
        
        gt_points = xyzs_neighbors[mask].reshape(batch_Length_Primitive*N_masked, self.tk, self.nsamples, 3)            # [B*M, tk, nn, 3]
        gt_points_list = torch.split(tensor=gt_points, split_size_or_sections=1, dim=1)
        gt_points_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in gt_points_list]           # tk*[B*M, nn, 3]

        point_loss = 0
        for tk_i in range(self.tk):
            point_loss += self.criterion_dist(pred_list[tk_i], gt_points_list[tk_i])
            point_loss = point_loss /1200#(150*8)
            #print(point_loss)
        point_loss = point_loss / self.tk
        
        """
        self-distilion
        """
        masked_features = emb_features[:,-N_masked:,:]
        masked_features = self.decoder_feature(masked_features.permute(0,2,1)).permute(0,2,1)

        fea_emb_full = fea_emb_full.reshape(batch_size,-1,C)
        fea_emb_full = fea_emb_full[mask].reshape(batch_Length_Primitive,N_masked,-1)
        fea_emb_full = fea_emb_full.detach()

        loss_mask = 1 - F.cosine_similarity(fea_emb_full, masked_features).mean()
        
        
        loss =loss_mask + loss_predict + point_loss 


        if self.vis:
            vis_points = xyzs_neighbors[~mask].reshape(batch_size, -1, self.tk, self.nsamples, 3) # [B, L*N-m, tk*nn, 3]
            pre_points = pre_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3)           
            gt_points = gt_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3)
            return pre_points, gt_points, vis_points, mask
        else:
            return loss


    def forward(self, clips,clips_aug=None):
        # [B, L, N, 3]
        if self.pretraining:
            emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors ,fea_emb_full,L,N= self.forward_encoder(clips,clips_aug)

            if self.vis:
                pre_points, gt_points, vis_points, mask = self.forward_decoder(emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors,fea_emb_full,L,N)
                return pre_points, gt_points, vis_points, mask
            else:
                loss = self.forward_decoder(emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors,fea_emb_full,L,N)
                return loss
        else:
            output = self.forward_encoder(clips,clips_aug)
            return output

    