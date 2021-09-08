import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'

        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        x = x.flatten(2).swapaxes(1, 2)     # b x (14*14) x hidden_dim
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=12):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor) -> Tensor:
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


vit_settings = {    
    'S/8': [8, 12, 384, 6],   #[patch_size, number_of_layers, embed_dim, heads]
    'S/16': [16, 12, 384, 6],
    'B/16': [16, 12, 768, 12]
}


class ViT(nn.Module):
    def __init__(self, model_name: str = 'S/8', image_size: int = 224) -> None:
        super().__init__()
        assert model_name in vit_settings.keys(), f"DeiT model name should be in {list(vit_settings.keys())}"
        patch_size, layers, embed_dim, heads = vit_settings[model_name]

        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, heads)
        for i in range(layers)])

        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(self, x: Tensor, W: int, H: int) -> Tensor:
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if num_patches == N and H == W:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = W // self.patch_size
        h0 = H // self.patch_size

        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic'
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def encode_image(self, x):
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        B, C, W, H = x.shape
        x = self.patch_embed(x)             
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.interpolate_pos_encoding(x, W, H)
        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]
        

if __name__ == '__main__':
    model = ViT('S/16')
    model.load_state_dict(torch.load('checkpoints/vit/dino_deitsmall16_pretrain.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)