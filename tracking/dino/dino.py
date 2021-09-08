import torch
from torchvision import transforms as T
from .xcit import XciT
from .vit import ViT

__all__ = ["load"]


def load(model_name, model_path, device):
    _, base_name, variant = model_name.split('-')
    model = eval(base_name)(variant)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, transform

