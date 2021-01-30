"""
Code modified from X. Wang et al., Enhanced Super-Resolution Generative Adversarial Networks,
The European Conference on Computer Vision Workshops (ECCVW), 2018
https://github.com/BlueAmulet/ESRGAN/blob/master/test.py
"""


def extract_model_parameters(state_dict):
    """
    Extracts model parameters from state dict loaded from PyTorch
    :param state_dict: Dict from torch.load()
    :return: Tuple containing ESRGAN model parameters
    """
    in_nc = None
    out_nc = None
    nf = None
    nb = None

    scale2 = 0
    max_part = 0
    for part in list(state_dict):
        parts = part.split('.')
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == 'sub':
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if part_num > 6 and parts[2] == 'weight':
                scale2 += 1
            if part_num > max_part:
                max_part = part_num
                out_nc = state_dict[part].shape[0]
    upscale = 2 ** scale2
    in_nc = state_dict['model.0.weight'].shape[1]
    nf = state_dict['model.0.weight'].shape[0]

    return in_nc, out_nc, nf, nb, upscale
