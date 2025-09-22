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
from modules.decoder_transformer import *
from models.model_utils import *


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
                 queue_size = 12288,
                 pretraining=True,
                 vis=False,
                 ):
        super(ContrastiveMaskedAutoEncoder, self).__init__()

        self.pretraining = pretraining

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[en_emb_dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        #self.motion = MotionEmbedding(spatial_kernel_size=[radius, nsamples])
        # encoder        
        self.encoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=en_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.encoder_transformer = Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
        self.encoder_norm = nn.LayerNorm(en_emb_dim)
        
        self.vis = vis
        self.nsamples = nsamples
        self.tk = temporal_kernel_size
        self.shotL = getDescriptorLength(elevation_divisions=2,
                                        azimuth_divisions=4
        )
        if self.pretraining:
            self.queue_size=queue_size#10752#msr:12288#ntu10752
            # 初始化动态队列
            self.register_buffer("queue", torch.randn(en_emb_dim, self.queue_size))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
            
            
            self.global_transformer=Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
            self.global_norm=nn.LayerNorm(en_emb_dim)
            # Mask
            self.mask_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
            trunc_normal_(self.mask_token, std=.02)

            self.feature_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
            trunc_normal_(self.feature_token, std=.02)


            # decoder
            self.decoder_embed = nn.Linear(en_emb_dim, de_emb_dim, bias=True)
            self.decoder_cl = nn.Linear(de_emb_dim, en_emb_dim, bias=True)

            self.decoder_frame = nn.Sequential(
                nn.Linear(de_emb_dim, 4*de_emb_dim, bias=True),
                nn.BatchNorm1d(4*de_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(4*de_emb_dim, en_emb_dim, bias=True),
                )#
            self.decoder_frame_backward = nn.Sequential(
                nn.Linear(de_emb_dim, 4*de_emb_dim, bias=True),
                nn.BatchNorm1d(4*de_emb_dim),
                nn.ReLU(inplace=True),
                #nn.Dropout(0.5),
                #nn.Linear(2*de_emb_dim, 4*de_emb_dim, bias=True),
                #nn.BatchNorm1d(4*de_emb_dim),
                #nn.ReLU(inplace=True),
                #nn.Dropout(0.5),
                nn.Linear(4*de_emb_dim, en_emb_dim, bias=True),
                )#
            
            self.decoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=de_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)
            
            self.decoder_transformer = Transformer(de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1)
            self.decoder_norm = nn.LayerNorm(de_emb_dim)

            # points_predictor
            self.points_predictor = nn.Conv1d(de_emb_dim, 3 * nsamples * temporal_kernel_size, 1)

            self.decoder_feature=nn.Conv1d(de_emb_dim,en_emb_dim,1)

            # loss
            self.criterion_dist = ChamferDistanceL2().cuda()
            self.criterion_cl = nn.CrossEntropyLoss().cuda()
            self.criterion_frame = nn.CrossEntropyLoss().cuda()
            
            self.mask_ratio = mask_ratio
            self.temperature=temperature
            self.momentum=momentum

        else:
            # PointMAE mlp_head
            self.cls_token_ft = nn.Parameter(torch.zeros(1, 1, en_emb_dim))
            self.cls_pos_ft = nn.Parameter(torch.randn(1, 1, en_emb_dim))
            
            trunc_normal_(self.cls_token_ft, std=.02)
            trunc_normal_(self.cls_pos_ft, std=.02)
            
            
            self.mlp_head = nn.Sequential(
                nn.Linear(en_emb_dim*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(256, num_classes)
            )
            """
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(en_emb_dim),
                nn.Linear(en_emb_dim, en_mlp_dim),
                nn.GELU(),
                #nn.Dropout(0.3),
                nn.Linear(en_mlp_dim, num_classes),
            )
            """
        # self.apply(self._init_weights)


    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def momentum_update(self):
        for param_b, param_m in zip(self.encoder_transformer.parameters(),
                                    self.global_transformer.parameters()):
            
            #print(f"param_b shape: {param_b.shape}, param_m shape: {param_m.shape}")

            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)
        """
        for param_b, param_m in zip(self.decoder_featureV2.parameters(),
                                    self.proj_head.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)
        """
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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # 获取队列位置
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0, "Queue size must be divisible by batch size"

        # 替换队列中的 keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # 更新队列指针

        self.queue_ptr[0] = ptr

    def forward_encoder(self, clips,clips_aug=None):
        # [B, L, N, 3]
        xyzs, features, xyzs_neighbors, shot_descriptors = self.tube_embedding(clips)
        features = features.permute(0, 1, 3, 2)                                              # [B, L, N, C]    
        batch_size, L, N, C = features.shape
        # xyzt position
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]                # L*[B, N, 3]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((batch_size, N, 1), dtype=torch.float32, device=clips.device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)                                            # [B, L, N, 4]
        #xyzts = torch.cat((xyzts,motion),dim=-1)                                             # [B, L, N, 4+3]
        # Token sequence
        #xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 7]
        #features = torch.reshape(input=features, shape=(batch_size, L*N, C))

        if self.pretraining:
            # Targets
            xyzs_neighbors = torch.reshape(input=xyzs_neighbors, shape=(batch_size, L* N, self.tk, self.nsamples, 3)) # [B, L*N, tk, nn, 3]
            shot_descriptors = torch.reshape(input=shot_descriptors, shape=(batch_size, L, N, self.tk, self.shotL))   # [B, L*N, tk, shotL]
            
            # Masking
            xyzts=xyzts.reshape(batch_size*L,N,4)
            bool_masked_pos = self.random_masking(xyzts)

            bool_masked_pos = bool_masked_pos.reshape(batch_size,-1)

            xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 7]
            features = torch.reshape(input=features, shape=(batch_size, L*N, C))
            # Encoding the visible part
            pos_emb_full = self.encoder_pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
            
            
            fea_emb_vis = features[~bool_masked_pos].reshape(batch_size, -1, C)
            pos_emb_vis = pos_emb_full[~bool_masked_pos].reshape(batch_size, -1, C)
            
            
            fea_emb_vis = fea_emb_vis + pos_emb_vis
            fea_emb_full = features + pos_emb_full
            
            fea_emb_vis = self.encoder_transformer(fea_emb_vis)
            fea_emb_vis = self.encoder_norm(fea_emb_vis)

            
            fea_emb_full = self.global_transformer(fea_emb_full)
            fea_emb_full = self.global_norm(fea_emb_full)
            
            return fea_emb_vis, bool_masked_pos, xyzts, xyzs_neighbors, shot_descriptors,fea_emb_full,L,N

        else:
            # Token sequence
            xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 4]
            features = torch.reshape(input=features, shape=(batch_size, L*N, C))
            xyzts = self.encoder_pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

            cls_tokens = self.cls_token_ft.expand(batch_size, -1, -1)
            cls_pos = self.cls_pos_ft.expand(batch_size, -1, -1)

            features = torch.cat((cls_tokens, features), dim=1)
            xyzts = torch.cat((cls_pos, xyzts), dim=1)

            embedding = xyzts + features

            output = self.encoder_transformer(embedding)
            output = self.encoder_norm(output)
            concat_f = torch.cat([output[:, 0], output[:, 1:].max(1)[0]], dim=-1)

            output = self.mlp_head(concat_f)

            return output              


    def forward_decoder(self, emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors,fea_emb_full,L,N):
        device = xyzts.get_device()
        
        
        emb_vis = self.decoder_embed(emb_vis)#b N_vis 512
        batch_size, N_vis, C_decoder = emb_vis.shape#b N_vis 512
        batch_size, N_full,C_encoder = fea_emb_full.shape#b N_full 1024
        emb_frame_encoder = emb_vis.reshape(batch_size,L,-1,C_decoder)#b L 8 512
        emb_frame_encoder = torch.max(input=emb_frame_encoder,dim=2,keepdim=False)[0]#B L 512
        pos_emd_vis = xyzts[~mask].reshape(batch_size, -1, 4)
        pos_emd_mask = xyzts[mask].reshape(batch_size, -1, 4)
        pos_emd_vis = self.decoder_pos_embed(pos_emd_vis.permute(0, 2, 1)).permute(0, 2, 1)
        pos_emd_mask = self.decoder_pos_embed(pos_emd_mask.permute(0, 2, 1)).permute(0, 2, 1)#b l*masked D
        
        _,N_masked,_ = pos_emd_mask.shape
        """
        video-level CL
        """
        emb_cl = self.decoder_cl(emb_vis)#b N_vis 1024
        emb_cl = torch.max(input=emb_cl,dim=1,keepdim=False)[0]#b 1024
        emb_cl = F.normalize(emb_cl,dim=-1)#b 1024
        
        emb_teacher = fea_emb_full.detach()
        emb_teacher_frame = emb_teacher.reshape(batch_size,L,-1,C_encoder)#b L 32 1024
        emb_teacher_frame = torch.max(input=emb_teacher_frame,dim=2,keepdim=False)[0]#b L 1024
        emb_teacher = torch.max(input=emb_teacher,dim=1,keepdim=False)[0]#b 1024
        emb_teacher = F.normalize(emb_teacher,dim=-1)
        
        
        positive_logits = torch.einsum('b c,b c->b ', [emb_cl, emb_teacher]).unsqueeze(-1)
        negative_logits = torch.einsum('b c,c k->b k', [emb_cl, self.queue.clone().detach()])
        logits = torch.cat([positive_logits, negative_logits], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss_global = self.criterion_cl(logits / self.temperature, labels)
        self._dequeue_and_enqueue(emb_teacher)
        
        
        loss_predict_forward = 0
        for b in range(batch_size):
            source_feature = emb_frame_encoder[b]#l 512
            target_feature = emb_teacher_frame[b]#l 1024

            predict_feature = self.decoder_frame(source_feature[:-1])#l-1 512 -> l-1 1024

            predict_feature_new = predict_feature
            key_feature_new = target_feature[1:]#l-1 1024

            predict_feature = torch.nn.functional.normalize(predict_feature_new, dim=1)
            key_feature = torch.nn.functional.normalize(key_feature_new, dim=1)

            logits = torch.mm(predict_feature, key_feature.transpose(1, 0))#[l-1,l-1]
            
            labels = torch.arange(key_feature.size()[0])
            labels = labels.cuda()
            loss_tmp = self.criterion_frame(logits / self.temperature, labels)

            loss_predict_forward = loss_tmp + loss_predict_forward

        loss_predict_forward = loss_predict_forward / batch_size
        """
        loss_predict_backward = 0
        for b in range(batch_size):
            source_feature = emb_frame_encoder[b]#l 512
            target_feature = emb_teacher_frame[b]#l 1024

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
        """
        
        #loss_predict = (loss_predict_forward+loss_predict_backward)*0.5
        loss_predict = loss_predict_forward
        mask_tokens = self.mask_token.expand(batch_size, N_masked, -1)
        feature_tokens = self.feature_token.expand(batch_size, N_masked, -1)

        emb_all = torch.cat([emb_vis, mask_tokens], dim=1)
        emb_all_feature = torch.cat([emb_vis, feature_tokens], dim=1)
        
        pos_all = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        emb_all = emb_all + pos_all
        emb_all_feature = emb_all_feature + pos_all

        emb_all = torch.cat((emb_all,emb_all_feature),dim=0)
        
        emb_all= self.decoder_transformer(emb_all)  
        emb_all = self.decoder_norm(emb_all)
        
        
        emb_points,emb_features = emb_all[0:batch_size,:,:],emb_all[batch_size:,:,:]
        
        
       
        masked_emb = emb_points[:, -N_masked:, :]       # [B, M, C]
        
        masked_emb = masked_emb.transpose(1, 2)      # [B, C, M]
        
        # reconstruct points
        pre_points = self.points_predictor(masked_emb).transpose(1, 2)   

        pre_points = pre_points.reshape(batch_size*N_masked, self.tk, self.nsamples, 3)                     # [B*M, tk, nn, 3]
        pred_list = torch.split(tensor=pre_points, split_size_or_sections=1, dim=1)     
        pred_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in pred_list] 
        
        # forward Loss
        
        gt_points = xyzs_neighbors[mask].reshape(batch_size*N_masked, self.tk, self.nsamples, 3)            # [B*M, tk, nn, 3]
        gt_points_list = torch.split(tensor=gt_points, split_size_or_sections=1, dim=1)
        gt_points_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in gt_points_list]           # tk*[B*M, nn, 3]

        point_loss = 0
        for tk_i in range(self.tk):
            point_loss += self.criterion_dist(pred_list[tk_i], gt_points_list[tk_i])
        point_loss = point_loss / self.tk
    
        masked_features = emb_features[:,-N_masked:,:]
        masked_features = self.decoder_feature(masked_features.permute(0,2,1)).permute(0,2,1)

        
        fea_emb_full = fea_emb_full[mask].reshape(batch_size,N_masked,-1)
        fea_emb_full = fea_emb_full.detach()

        loss_mask = 1 - F.cosine_similarity(fea_emb_full, masked_features).mean()
        
        
        loss =loss_mask  + loss_global + loss_predict + point_loss 
        
        

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

    