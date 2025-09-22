from transformers import AutoTokenizer, XCLIPTextModel
import torch.nn as nn
import torch
import timm
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 XCLIP 模型和分词器
        self.model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        
        # 冻结 XCLIP 模型的参数
        for p in self.model.parameters():
            p.requires_grad = False

        self.template = ['']
        self.template_last = ['.']

    def forward(self, text, device,index=0):
        # 将 text 转换为张量
        if isinstance(text, str):
            prompt_text = text
        else:
            prompt_text = [random.choice(self.template) + t + random.choice(self.template_last) for t in text]

        # 对 text 进行分词并生成张量
        prompt_text = self.tokenizer(prompt_text, padding=True, return_tensors="pt").to(device)
        self.model = self.model.to(device)
        # 将生成的张量移动到指定的设备
        #prompt_text = {key: value.to(device) for key, value in prompt_text.items()}
        
        # 通过 XCLIP 模型得到特征
        outputs = self.model(**prompt_text)
        text_feature = outputs.pooler_output  # 获取池化的特征

        return text_feature

    
if __name__ == '__main__':
    text='High Arm Wave'
    # Initialize the attention layers with correct dimensions
    textEncoder = TextEncoder().cuda()
    text_feature = textEncoder(text)
    print(text_feature)