from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector
import kornia

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    
    for key, val in batch_dict.items():
        if key == 'TTA_data_dict_list':
            for tta_idx in range(len(val)):
                load_data_to_gpu(val[tta_idx])
            batch_dict[key] = val
        elif not isinstance(val, np.ndarray) or key in ['TTA_trans_dict', 'gt_names', 'gt_dense']:
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            if val.ndim == 5:
                imgs = val
                imgs = imgs.reshape((imgs.shape[0] * 5, imgs.shape[2], imgs.shape[3], imgs.shape[4]))   
                imgs = kornia.image_to_tensor(imgs).float().cuda().contiguous() # (BN, C, H, W)
                imgs = imgs.view(-1, 5, imgs.shape[1], imgs.shape[2], imgs.shape[3])
                batch_dict[key] = imgs
            else:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
