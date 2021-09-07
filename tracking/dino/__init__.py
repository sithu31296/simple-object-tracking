import torch
from torchvision import transforms as T
from .xcit import XciT


def load(model_path, device):
    model = XciT('S12/8', image_size=224)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    transform = T.Compose(
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    return model, transform

