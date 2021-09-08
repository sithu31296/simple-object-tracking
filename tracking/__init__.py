import os
from .utils import download
from .clip import load as clip_load
from .dino import load as dino_load


_MODELS = {
    "CLIP-RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "CLIP-ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "DINO-XciT-S12/16": "https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth",
    "DINO-XciT-M24/16": "https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
    "DINO-ViT-S/16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    "DINO-ViT-B/16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
}


def load_feature_extractor(model_name: str, device):
    assert model_name in _MODELS
    model_path = download(_MODELS[model_name], os.path.expanduser("~/.cache/tracking"))

    if model_name.startswith('CLIP'):
        model, transform = clip_load(model_path, device, jit=False)
    elif model_name.startswith('DINO'):
        model, transform = dino_load(model_name, model_path, device)
    return model, transform

