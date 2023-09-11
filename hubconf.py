dependencies = ['torch', 'torchvision']
from models.our_detector import Our
import torch

# Detector is the name of entrypoint
def Detector(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    PRL 2023 Detector model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights

    if pretrained:
        model = Our(enc_channels = [1, 32, 64, 128], deformable_encoder = False, deformable_decoder = False)
        checkpoint = 'https://github.com/verlab/LearningToDetect_PRL_2023/raw/main/pretrained/our/final_model_1500.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, map_location=torch.device("cpu")))
    else:
        model = Our(**kwargs)

    return model