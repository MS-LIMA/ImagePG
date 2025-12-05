import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from kornia import normalize
except:
    pass
from collections import OrderedDict
from ..model_utils.basic_block_2d import BasicBlock2D
from .swin import SwinTransformer
from .img_neck.generalized_lss import GeneralizedLSSFPN

class MMDETFPNKITTI(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.imgconfig = model_cfg.IMGCONFIG
        self.fpnconfig = model_cfg.FPNCONFIG
        self.img_backbone = SwinTransformer(self.imgconfig)
        if self.model_cfg.get('PRETRAINEDPATH', None) is not None:
            self.img_backbone.init_weights(self.model_cfg.PRETRAINEDPATH)
        if self.imgconfig.get('pretrained', None) is not None: 
            self.img_backbone.init_weights(self.imgconfig.pretrained)

        self.neck = GeneralizedLSSFPN(self.fpnconfig)
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.IFN.CHANNEL_REDUCE["in_channels"]):
            _channel_out = model_cfg.IFN.CHANNEL_REDUCE["out_channels"][_idx]
            self.out_channels[model_cfg.IFN.ARGS['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {"in_channels": _channel,
                         "out_channels": _channel_out,
                         "kernel_size": model_cfg.IFN.CHANNEL_REDUCE["kernel_size"][_idx],
                         "stride": model_cfg.IFN.CHANNEL_REDUCE["stride"][_idx],
                         "bias": model_cfg.IFN.CHANNEL_REDUCE["bias"][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))

        self.init_weights()

        self.freeze_img = model_cfg.IMAGE_BACKBONE.get('FREEZE_IMGBACKBONE', False)
        if self.freeze_img:
            self.freeze()

    def freeze(self):
        if self.freeze_img:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

            for param in self.neck.parameters():
                param.requires_grad = False        

    def init_weights(self):
        model_cfg = self.model_cfg
        if 'IMGPRETRAINED_MODEL' in model_cfg:
            checkpoint= torch.load(model_cfg.IMGPRETRAINED_MODEL, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            ckpt = state_dict
            new_ckpt = OrderedDict()
            import pdb;pdb.set_trace()
            for k, v in ckpt.items():
                if k.startswith('backbone'):
                    new_v = v
                    new_k = k.replace('backbone.', 'img_backbone.')
                else:
                    continue
                new_ckpt[new_k] = new_v
            self.img_backbone.load_state_dict(new_ckpt, strict=False)

    def get_output_feature_dim(self):
        return self.out_channels

    def forward(self, batch_dict):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """

        # Extract features
        result = OrderedDict()
        images = batch_dict['images']
        bs = batch_dict['batch_size']
        batch_dict['image_features'] = {}
        single_result = {}
        B, C, H, W = images.shape
        x = self.img_backbone(images)
        x_neck = self.neck(x)
        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            image_features = x_neck[_idx]
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)
            single_result[_layer+"_feat2d"] = image_features
        for layer in single_result.keys():
            if layer not in batch_dict['image_features'].keys():
                batch_dict['image_features'][layer] = {}
            batch_dict['image_features'][layer]= single_result[layer]
        return batch_dict

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        # if self.pretrained:
        #     # Create a mask for padded pixels
        #     mask = torch.isnan(x)

        #     # Match ResNet pretrained preprocessing
        #     x = normalize(x, mean=self.norm_mean, std=self.norm_std)

        #     # Make padded pixels = 0
        #     x[mask] = 0

        return x