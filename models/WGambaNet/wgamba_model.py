from .wgamba import WGamba
import torch
from torch import nn


class WgambaNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=8,
                 depths=None,
                 depths_decoder=None,
                 drop_path_rate=0.,
                 load_ckpt_path=None,
                 d_state=16,
                 dims=None,
                 dims_decoder=None,
                 ):
        super().__init__()
        if dims_decoder is None:
            dims_decoder = [768, 384, 192, 96]
        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [2, 2, 9, 2]
        if depths_decoder is None:
            depths_decoder = [2, 9, 2, 2]
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.wgambanet = WGamba(in_chans=input_channels,
                                num_classes=num_classes,
                                depths=depths,
                                depths_decoder=depths_decoder,
                                drop_path_rate=drop_path_rate,
                                d_state=d_state,
                                dims=dims,
                                dims_decoder=dims_decoder,
                                )

    def forward(self, x, valid_mask=None):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if valid_mask is not None:
            logits = self.wgambanet(x, valid_mask=valid_mask)
        else:
            logits = self.wgambanet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            # load encoder weights
            model_dict = self.wgambanet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu')
            pretrained_dict = modelCheckpoint['model']
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)))
            self.wgambanet.load_state_dict(model_dict, strict=False)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            # load decoder weights（encoder -> decoder）
            model_dict = self.wgambanet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu')
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)))
            self.wgambanet.load_state_dict(model_dict, strict=False)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")

            # ASDG VSSBlock load
            cross_vss_mappings = [
                ('asdg_inx_2.vss_blocks.0', 'layers.2.blocks.0'),
                ('asdg_inx_2.vss_blocks.1', 'layers.2.blocks.1'),
                ('asdg_inx_3.vss_blocks.0', 'layers.1.blocks.0'),
                ('asdg_inx_3.vss_blocks.1', 'layers.1.blocks.1'),
            ]
            # Reload the latest state_dict once
            model_dict = self.wgambanet.state_dict()
            pretrained_dict = modelCheckpoint['model']

            for cross_prefix, main_prefix in cross_vss_mappings:
                for name in model_dict.keys():
                    if name.startswith(cross_prefix):
                        main_key = name.replace(cross_prefix, main_prefix)
                        if main_key in pretrained_dict:
                            model_dict[name] = pretrained_dict[main_key]
            self.wgambanet.load_state_dict(model_dict, strict=False)
            print('ASDG VSSBlock loaded finished！')
