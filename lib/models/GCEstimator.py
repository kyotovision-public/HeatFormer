import torch.nn as nn
import torch

from einops import rearrange
from lib.models.pose_Transformer import TransformerDecoder

class G_HUMANS(nn.Module):
    def __init__(self):
        super().__init__()

        decoder_args = dict(
            num_tokens=1,
            token_dim=1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            context_dim=1280
        )

        self.decoder = TransformerDecoder(**decoder_args)
        self.decpose = nn.Linear(1024, 24*6)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')

        batch = x.shape[0]
        token = torch.zeros(batch, 1, 1).to(x.device)

        out = self.decoder(token, context=x).squeeze()

        pose = self.decpose(out)
        # shape = self.decshape(out)
        cam = self.deccam(out)

        return pose[:, :6], cam