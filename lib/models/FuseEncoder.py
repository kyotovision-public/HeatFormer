import torch 
import torch.nn as nn

from einops import rearrange

from lib.models.pe import SinePositionalEncoding3D
from lib.models.Transformer import TransformerEncoder

class FuseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_dim = 16 * 16

        transformer_ecoder_args = dict(
            num_tokens=17 + 1,
            token_dim=self.token_dim,
            dim=1024,
            depth=cfg.MODEL.ENCODER.DEPTH,
            heads=cfg.MODEL.ENCODER.HEADS,
            mlp_dim=cfg.MODEL.ENCODER.DIM,
            pos_embed=True, # <- joints pos embed
            pos_embed_dim=True
        )

        self.Encoder = TransformerEncoder(**transformer_ecoder_args)
        self.n_views = cfg.DATASET.NUM_VIEWS

    def forward(self, heatmap, debug=False):
        #  heatmap : (batch * n_view, joints, h, w)
        #                  □□□
        # batch * joints * □□□ => (batch * H * W) * joints * □
        #                  □□□

        heatmap_1 = rearrange(heatmap, "b j (h_1 h_2) (w_1 w_2) -> (b h_1 w_1) j (h_2 w_2)", h_1=16, w_1=12)
        # if debug:
        #     db = rearrange(heatmap_1, '(b h1 w1) j (h2 w2) -> b h1 w1 j h2 w2', h1=16,w1=12,h2=16)
        #     heatmap_debug = torch.zeros(17, 256, 192)
        #     for hh in range(16):
        #         for ww in range(12):
        #             for jj in range(17):
        #                 heatmap_debug[jj, 16*hh:16*(hh+1), 16*ww:16*(ww+1)] = db[0, hh, ww, jj]
        #     return heatmap_debug.max(dim=0)[0]

        batch = heatmap_1.shape[0]

        token = torch.zeros(batch, 1, self.token_dim).to(self.cfg.DEVICE)

        inp = torch.cat([heatmap_1, token], dim=1)  # ((batch h w) joints + 1 C)

        token_out = self.Encoder(inp)[:, -1].squeeze()

        token_out = rearrange(token_out, "(b h w) c -> b c h w", h=16, w=12)  # (batch * n_view 1024 16 12)
        
        return token_out
    
class FuseEncoder_DINO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_dim = 14 * 14

        transformer_ecoder_args = dict(
            num_tokens=17 + 1,
            token_dim=self.token_dim,
            dim=1024,
            depth=cfg.MODEL.ENCODER.DEPTH,
            heads=cfg.MODEL.ENCODER.HEADS,
            mlp_dim=cfg.MODEL.ENCODER.DIM,
            pos_embed=True # <- joints pos embed
        )

        self.Encoder = TransformerEncoder(**transformer_ecoder_args)
        self.n_views = cfg.DATASET.NUM_VIEWS

    def forward(self, heatmap):
        #  heatmap : (batch n_view, joints, h, w)
        heatmap_1 = rearrange(heatmap, "b j (h_1 h_2) (w_1 w_2) -> (b h_2 w_2) j (h_1 w_1)", h_1=14, w_1=14)

        batch = heatmap_1.shape[0]

        token = torch.zeros(batch, 1, self.token_dim).to(self.cfg.DEVICE)

        inp = torch.cat([heatmap_1, token], dim=1)  # ((batch h w) joints + 1 C)

        token_out = self.Encoder(inp)[:, -1].squeeze()

        token_out = rearrange(token_out, "(b h w) c -> b c h w", h=16, w=16)  # (batch * n_view 1024 16 12)
        
        return token_out