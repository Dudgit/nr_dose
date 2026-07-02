import pytorch_lightning as pl
import torch
import torch.nn as nn

from monai.networks.blocks import UnetResBlock
from monai.networks.nets import ViT
from monai.networks.blocks import UnetResBlock

class Encoder(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64, 128, 256)):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c in channels:
            blocks.append(UnetResBlock(spatial_dims=3,in_channels=c_in,out_channels=c,kernel_size=3,stride=2,norm_name="INSTANCE",))
            c_in = c
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)
    
class Decoder(nn.Module):
    def __init__(self, channels=(256, 128, 64, 32), out_channels=1):
        super().__init__()

        layers = []

        # This upsamples 3 times: 256->128, 128->64, 64->32
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2),
                nn.InstanceNorm3d(cout),
                nn.GELU(),
            ])

        # ADD THE 4TH UPSAMPLE STEP HERE: 32->32
        # This restores the final missing spatial dimension
        last_channel = channels[-1]
        layers.extend([
            nn.ConvTranspose3d(last_channel, last_channel, kernel_size=2, stride=2),
            nn.InstanceNorm3d(last_channel),
            nn.GELU(),
        ])

        self.decoder = nn.Sequential(*layers)
        self.out = nn.Conv3d(last_channel, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(self.decoder(x))
    

class FourierEmbedding(nn.Module):

    def __init__(self, num_frequencies=8):
        super().__init__()

        freqs = 2 ** torch.arange(num_frequencies).float() * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = x[..., None] * self.freqs
        return torch.cat(
            [torch.sin(x), torch.cos(x)],
            dim=-1
        ).flatten(1)
    

class ConditionedTransformer(nn.Module):
    def __init__(self,embed_dim=256,num_heads=8,num_layers=4,mlp_ratio=4,):
        super().__init__()

        # Beam encoder
        self.energy_mlp = nn.Sequential(nn.Linear(1, 64),nn.GELU())
        self.condition_proj = nn.Sequential(nn.Linear(128,embed_dim),nn.GELU())
        self.coord_encoder = nn.Sequential(nn.Linear(3, 64),nn.GELU(),nn.Linear(64,64))
        #FourierEmbedding(num_frequencies=8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,)

        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=num_layers,)

    def forward(self, x, beam):
        """
        x    : (B,C,D,H,W)
        beam : (B,4)
        """

        B, C, D, H, W = x.shape

        # (B,N,C)
        x = x.flatten(2).transpose(1, 2)

        # (Connditioning)
        xyz = self.coord_encoder(beam[:, :3])
        energy = self.energy_mlp(beam[:, 3:])
        condition = torch.cat([xyz, energy], dim=-1)
        condition = self.condition_proj(condition)  # (B,1,C)


        # Add conditioning to every token
        x = x + condition.unsqueeze(1)

        x = self.transformer(x)

        # back to volume
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x

class DoTA_based(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=(32, 64, 128, 256), num_heads=8, num_layers=4):
        super().__init__()
        embed_dim = channels[-1]
        self.encoder = Encoder(in_channels=in_channels, channels=channels)
        self.transformer = ConditionedTransformer(embed_dim=embed_dim,num_heads=num_heads,num_layers=num_layers,)
        self.decoder = Decoder(channels=list(reversed(channels)), out_channels=out_channels)

    def forward(self, x, beam):
        x = self.encoder(x)
        x = self.transformer(x, beam)
        x = self.decoder(x)
        return x