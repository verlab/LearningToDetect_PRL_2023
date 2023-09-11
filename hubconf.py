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
    model = Our(**kwargs)

    if pretrained:
        checkpoint = 'https://github.com/verlab/LearningToDetect_PRL_2023/raw/main/pretrained/our/final_model_1500.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))

    return model