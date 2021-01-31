"""
Code modified from X. Wang et al., Enhanced Super-Resolution Generative Adversarial Networks,
The European Conference on Computer Vision Workshops (ECCVW), 2018
https://github.com/BlueAmulet/ESRGAN/blob/master/architecture.py
"""

import math
import torch.nn as nn
from src.ESRGAN import block


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA',
                 res_scale=1, upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = block.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [block.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        lr_conv = block.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = block.upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = block.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        hr_conv0 = block.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        hr_conv1 = block.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = block.sequential(fea_conv, block.ShortcutBlock(block.sequential(*rb_blocks, lr_conv)), *upsampler,
                                      hr_conv0,
                                      hr_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
