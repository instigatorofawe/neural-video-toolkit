def extract_model_parameters(state_dict):
    """
    Extracts model parameters from saved PyTorch state_dict
    :param state_dict: Result from torch.load()
    :return: Scale factor of pretrained model
    """
    if state_dict['module.tail.0.0.weight'].shape[0] == 2304:
        scale = 3
    elif 'module.tail.0.2.weight' in state_dict.keys():
        scale = 4
    else:
        scale = 2
    return scale
