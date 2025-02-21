import torch
import torch.nn as nn
from einops import rearrange

class AdaLayerNorm(nn.Module):
    """borrow from PMCE"""
    def __init__(self, dim, norm_cond_dim, esp=1e-6):
        super().__init__()

        assert dim>0
        assert norm_cond_dim>0

        self.gamma_mlp = nn.Linear(norm_cond_dim, dim)
        self.beta_mlp = nn.Linear(norm_cond_dim, dim)
        self.esp = esp
    
    def forward(self, x, t):
        """
        x : (batch ... dim)
        t : (batch norm_cond_dim)
        return : (batch dim)
        """
        size = x.size()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        gamma = self.gamma_mlp(t).view(size[0], 1, -1).expand(size)
        beta = self.beta_mlp(t).view(size[0], 1, -1).expand(size)

        return gamma * (x-mean) / (std + self.esp) + beta

class Add_Norm(nn.Module):
    def __init__(self, dim, f, norm_dim):
        super().__init__()
        # self.norm = AdaLayerNorm(dim, norm_dim)
        self.norm = nn.LayerNorm(dim)
        self.f = f
    
    def forward(self, x, *args, **kwargs):
        if isinstance(self.norm, AdaLayerNorm):
            return self.f(self.norm(x, *args), **kwargs)
        else:
            return self.f(self.norm(x), **kwargs)

class FF(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layer(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()

        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.x_to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.x_to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.x_to_v = nn.Linear(dim, hidden_dim, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.out_layer = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        # x : (batch num_token dim)
        q = rearrange(self.x_to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.x_to_k(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.x_to_v(x), "b n (h d) -> b h n d", h=self.heads)

        # calculate attn
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)

        output = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")

        return self.dropout(self.out_layer(output))

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, context_dim=None) -> None:
        super().__init__()

        hidden_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.x_to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.x_to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.x_to_v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout()

        context_dim = dim if context_dim is None else context_dim

        self.out_layer = nn.Linear(hidden_dim, dim)
    
    def forward(self, x, context=None):
        context = x if context is None else context

        q = rearrange(self.x_to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.x_to_k(context), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.x_to_v(context), "b n (h d) -> b h n d", h=self.heads)

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)

        output = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")

        return self.dropout(self.out_layer(output))

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        norm_cond_dim=-1
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout
            )

            ff = FF(
                input_dim=dim,
                hidden_dim=mlp_dim,
                dropout=dropout
            )

            self.layers.append(
                nn.ModuleList([
                    Add_Norm(dim, sa, norm_dim=norm_cond_dim),
                    Add_Norm(dim, ff, norm_dim=norm_cond_dim)
                ])
            )
    
    def forward(self, x, *args):
        for attn, ff in self.layers:
            x = x + attn(x, *args)
            x = x + ff(x, *args)
        return x

class TransformerCA(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        norm_cond_dim=-1,
        context_dim=None,
        self_attention=True
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout
            )

            ca = CrossAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                context_dim=context_dim
            )

            ff = FF(
                input_dim=dim,
                hidden_dim=mlp_dim,
                dropout=dropout
            )

            if self_attention:
                self.layers.append(
                    nn.ModuleList([
                        Add_Norm(dim, sa, norm_dim=norm_cond_dim),
                        Add_Norm(dim, ca, norm_dim=norm_cond_dim),
                        Add_Norm(dim, ff, norm_dim=norm_cond_dim)
                    ])
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        Add_Norm(dim, ca, norm_dim=norm_cond_dim),
                        Add_Norm(dim, ff, norm_dim=norm_cond_dim)
                    ])
                )

    
    def forward(self, x, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        assert len(context_list) == len(self.layers), f'list size is different : {len(context_list)} != {len(self.layers)}'

        for i, (sa_attn, ca_attn, ff) in enumerate(self.layers):
            x = x + sa_attn(x, *args)
            x = x + ca_attn(x, *args, context=context_list[i])
            x = x + ff(x, *args)
        return x
    
class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(
                    p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        token_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        norm_cond_dim=-1,
        pos_embed=False,
        pos_embed_dim=True
    ):
        super().__init__()

        self.to_token_embedding = nn.Linear(token_dim, dim)
        if pos_embed:
            if pos_embed_dim:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))  # 適切な位置埋め込みに変える
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, 1))
        self.dropout = DropTokenDropout(emb_dropout)

        self.Transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_cond_dim=norm_cond_dim
        )

        self.pos_embed = pos_embed
    
    def forward(self, x, *args, **kwargs):
        x = self.to_token_embedding(x)
        x = self.dropout(x)
        if self.pos_embed:
            x = x + self.pos_embedding[:, :x.shape[1]]

        x = self.Transformer(x, *args)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        token_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        norm_cond_dim=-1,
        context_dim=None,
        pos_embed=False,
        self_attention=True
    ):
        super().__init__()

        self.to_token_embedding = nn.Linear(token_dim, dim)
        if pos_embed:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.dropout = DropTokenDropout(emb_dropout)

        self.transformer = TransformerCA(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
            self_attention=self_attention
        )

        self.pos_embed = pos_embed
    
    def forward(self, x, *args, context=None, context_list=None):
        x = self.to_token_embedding(x)
        x = self.dropout(x)
        if self.pos_embed:
            x = x + self.pos_embedding[:, :x.shape[1]]

        x = self.transformer(x, *args, context=context, context_list=context_list)

        return x
