import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
 
class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [self.model_cfg.get("INPUT_CHANNELS", input_channels), *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
 
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        
        self.fuse_hm = self.model_cfg.get('FUSE_HM', False)
        if self.fuse_hm:
            from ..fusion_modules.adf_block import ADF
            c = self.model_cfg.get("INPUT_CHANNELS", input_channels)
            # c = sum(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            
            self.bev_fuse_blocks = []
            channels = [c + 1] + \
                self.model_cfg.FUSE_HM_FILTERS + [c]
            for i in range(len(channels) - 1):
                self.bev_fuse_blocks.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
                self.bev_fuse_blocks.append(nn.BatchNorm2d(channels[i+1], eps=0.001, momentum=0.01))
                self.bev_fuse_blocks.append(nn.ReLU())
            self.bev_fuse_blocks = nn.Sequential(*self.bev_fuse_blocks)
            self.ADF_blocks = ADF(c)
            
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
 
        if 'pillar_spatial_features' in data_dict.keys() and self.fuse_hm == False:
            spatial_features = data_dict['pillar_spatial_features']
        else:
            spatial_features = data_dict['spatial_features']
 
        ups = []
        ret_dict = {}
        x = spatial_features

        if self.fuse_hm and 'hm_prob' in data_dict.keys():
            hm = data_dict['hm_prob']
            x = torch.cat((x, hm), dim=1)
            x = self.bev_fuse_blocks(x)
            x = self.ADF_blocks(x)
            data_dict['ADF_features_2d'] = x   

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        if 'pillar_spatial_features' in data_dict.keys() and self.fuse_hm == False:
            data_dict.pop('pillar_spatial_features', None)
            data_dict['pillar_spatial_features_2d'] = x
        else:
            data_dict['spatial_features_2d'] = x
            
            # batch_size = data_dict['batch_size']
            # feature_map_np = x.cpu().detach()
            # for i in range(batch_size):
            #     feat = feature_map_np[i]
            #     heatmap = feat.softmax(dim=0).max(dim=0)[0].numpy()
            #     # heatmap 시각화
            #     plt.figure(figsize=(6, 6))
            #     plt.imshow(heatmap, interpolation='nearest')
            #     plt.colorbar(label='Activation')
            #     plt.title('2D Feature Map Heatmap')
            #     plt.xlabel('Width')
            #     plt.ylabel('Height')
            #     plt.show()
            
        
        return data_dict
