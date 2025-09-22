import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

class ContrastiveLearningDataset(Dataset):
    def __init__(self, root, frames_per_clip=16, step_between_clips=1, num_points=2048, sub_clips=5, train=True):
        super(ContrastiveLearningDataset, self).__init__()

        self.sub_clips = sub_clips

        self.videos = []
        self.labels = []
        self.index_map = []
        self.text=[]
        index = 0
        #Label_Action = {'0':'',   '1':'', '2':'', '3':'', '4':'', '5':'', '6':'', '7':'', '8':'', '9':'', '10':'', '11':'', '12':'', '13':'', '14':'',  '15':'', '16':'', '17':'', '18':'', '19':'',
        #}
        labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',  '15', '16', '17', '18', '19']
        Action = ['Waving High Arm', 'Waving Horizontal Arm', 'Hammering', 'Catching Hand', 'Punching Forward',
          'Throwing High', 'Drawing X', 'Drawing Tick', 'Drawing Circle', 'Clapping Hands',
          'Waving Two Hands', 'Boxing Side', 'Bending', 'Kicking Forward', 'Kicking Side',
          'Jogging', 'Swinging Tennis', 'Serving Tennis', 'Swinging Golf', 'Picking Up & Throwing']

        Label_Action = dict(zip(labels, Action))
        #print(Label_Action)
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                #text = int(video_name.split('_')[0].split('a')[0])
                
                text = Label_Action[f'{label}']
                #print(text)
                self.labels.append(label)
                self.text.append(text)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            # if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
            #     video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
            #     self.videos.append(video)
            #     label = int(video_name.split('_')[0][1:])-1
            #     self.labels.append(label)

            #     nframes = video.shape[0]
            #     for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
            #         self.index_map.append((index, t))
            #     index += 1

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]
        video = self.videos[index]
        text = self.text[index]
        clip = [video[t+i*self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        #clipv2 = copy.deepcopy(clip)
        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip.astype(np.float32) / 300 # [L, N, 3]


        # V2
        #scalesv2 = np.random.uniform(0.9, 1.1, size=3)
        #clipv2 = clipv2 * scalesv2
        #clipv2 = clipv2 / 300 

        #jittered_data = np.random.normal(0, 0.01, size=(clipv2.shape[0],clipv2.shape[1],3)).clip(-0.02, 0.02)
        #translation = np.random.normal(0, 0.01, size=(3)).clip(-0.05, 0.05)
        #clipv2 = clipv2 + jittered_data + translation
        #clipv2 = clipv2.astype(np.float32) # [L, N, 3]
        #clips = np.split(clip, indices_or_sections=self.sub_clips, axis=0) # S*[L', N, 3]
        #clips = torch.tensor(np.array(clips)) # [S, L', N, 3]

        return clip, text,index
            
        
if __name__ == '__main__':
    np.random.seed(0)
    dataset = ContrastiveLearningDataset(root='/root/P4Transformer-ours/data/msr_action')
    clips,text, video_index = dataset[0]
    print('clips.shape:', clips.shape)
    print('action class:', text)
    print(len(dataset))
