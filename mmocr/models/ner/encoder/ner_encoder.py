import torch.nn as nn
from mmcv.cnn import xavier_init

from mmocr.models.builder import ENCODERS


import torch.nn as nn
import torch
from mmocr.models.builder import ENCODERS
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import uniform_init, xavier_init

from mmdet.models.builder import BACKBONES
import sys
import math
import json

from .config import Config
from .bert import BertModel



@BACKBONES.register_module()
class BertSoftmaxForNer(nn.Module):

    # def __init__(self,config):
    def __init__(self):
        super().__init__()
        self.num_labels = 34
        self.config = Config()
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 34)
        self.loss_type = 'ce'
        self.init_weights()


    def forward(self,img_metas):
        input_ids=[]
        labels = []
        attention_masks = []
        token_type_ids=[]
        for i in range(len(img_metas)):
            input_id = torch.tensor(img_metas[i]["input_ids"]).cuda()
            label = torch.tensor(img_metas[i]["labels"]).cuda()
            attention_mask = torch.tensor(img_metas[i]["attention_mask"]).cuda()
            token_type_id = torch.tensor(img_metas[i]["token_type_ids"]).cuda()

            input_ids.append(input_id)
            labels.append(label)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        input_ids = torch.stack(input_ids,0)
        labels = torch.stack(labels,0)
        attention_masks = torch.stack(attention_masks, 0)
        token_type_ids = torch.stack(token_type_ids,0)

        outputs = self.bert(input_ids = input_ids,attention_mask=attention_masks,token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here


        softmax = F.softmax(outputs[0], dim=2)
        preds = softmax.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        return logits,preds


    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)


