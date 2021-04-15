from mmdet.models.builder import DETECTORS, build_backbone, build_loss
# from mmocr.models.builder import (build_convertor, build_decoder,
#                                   build_encoder, build_preprocessor)
from mmocr.models.textrecog.recognizer.base import BaseRecognizer
import torch.nn as nn
from mmocr.models.ner.loss.ner_loss import NerLoss
import torch

@DETECTORS.register_module()
class NerClassifier(BaseRecognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self, backbone=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.loss = NerLoss()

    def extract_feat(self, imgs):
        """Extract features from images."""
        x = self.backbone(imgs)

        return x

    def forward_train(self, imgs, img_metas, **kwargs):
        logits,x = self.backbone(img_metas)

        labels = []
        attention_masks = []
        for i in range(len(img_metas)):
            label = torch.tensor(img_metas[i]["labels"]).cuda()
            attention_mask = torch.tensor(img_metas[i]["attention_mask"]).cuda()
            labels.append(label)
            attention_masks.append(attention_mask)

        labels = torch.stack(labels, 0)
        attention_masks = torch.stack(attention_masks, 0)
        loss = self.loss(logits,labels,attention_masks)

        return {"loss_cls":loss}



    def forward_test(self, imgs, img_metas, **kwargs):
        logits,x = self.backbone(img_metas)
        return x

        # return self.simple_test(imgs, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]])
        """
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass

