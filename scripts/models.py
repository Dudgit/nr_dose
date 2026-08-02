import pytorch_lightning as pl
import torch
import torch.nn as nn

from monai.networks.blocks import UnetResBlock
from monai.networks.nets import ViT
from monai.networks.blocks import UnetResBlock
from scripts.metaembedder import FourierFeatureEmbedder

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
        return nn.functional.softplus(self.out(self.decoder(x)))

class FiLMLayer(nn.Module):
    def __init__(self, condition_dim, num_channels):
        super().__init__()
        # Project the 1D condition vector into 2 values (gamma, beta) for every channel
        self.proj = nn.Linear(condition_dim, num_channels * 2)
        
        # Initialize the projection so it starts as an identity mapping (safe starting point)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, condition):
        # x shape: [Batch, Channels, Z, Y, X]
        # condition shape: [Batch, condition_dim]
        
        # 1. Get gamma and beta
        params = self.proj(condition) # Shape: [Batch, Channels * 2]
        
        # 2. Reshape for 3D broadcasting
        params = params.view(params.size(0), params.size(1), 1, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        
        # 3. Apply modulation 
        # (1.0 + gamma) is a standard stabilizing trick so initialization equals identity
        return x * (1.0 + gamma) + beta

class FiLMDecoderBlock(nn.Module):
    def __init__(self, cin, cout, condition_dim):
        super().__init__()
        # 1. Expand spatial dimensions
        self.upsample = nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2)
        
        # 2. Refine and inject condition
        self.conv1 = nn.Conv3d(cout, cout, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(cout, affine=False) # Affine=False because FiLM handles scaling
        self.film1 = FiLMLayer(condition_dim, cout)
        self.act1 = nn.GELU()
        
        # 3. Optional extra capacity
        self.conv2 = nn.Conv3d(cout, cout, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(cout, affine=False)
        self.film2 = FiLMLayer(condition_dim, cout)
        self.act2 = nn.GELU()

    def forward(self, x, condition):
        x = self.upsample(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, condition) # Latent skip connection 1
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, condition) # Latent skip connection 2
        x = self.act2(x)
        
        return x


class Decoder_v2_MultiFiLM(nn.Module):
    def __init__(self, condition_dim, channels=(256, 128, 64, 32), out_channels=1):
        super().__init__()

        self.blocks = nn.ModuleList()
        
        # 3 Upsampling steps: 256->128, 128->64, 64->32
        for cin, cout in zip(channels[:-1], channels[1:]):
            self.blocks.append(FiLMDecoderBlock(cin, cout, condition_dim))

        # 4th step: 32->32
        last_channel = channels[-1]
        self.blocks.append(FiLMDecoderBlock(last_channel, last_channel, condition_dim))

        self.out = nn.Conv3d(last_channel, out_channels, kernel_size=1)

    def forward(self, x, condition):
        # Pass the condition vector directly into every spatial scale
        for block in self.blocks:
            x = block(x, condition)
            
        return nn.functional.softplus(self.out(x))


class Decoder_v2(nn.Module):
    def __init__(self, channels=(256, 128, 64, 32), out_channels=1):
        super().__init__()

        layers = []

        # This upsamples 3 times: 256->128, 128->64, 64->32
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers.extend([
                # 1. Expand spatial dimensions
                nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2),
                # 2. Refine and blend features (The missing piece!)
                nn.Conv3d(cout, cout, kernel_size=3, padding=1),
                nn.InstanceNorm3d(cout),
                nn.GELU(),
                # Optional: A second Conv3d here is very common in U-Nets for extra capacity
                nn.Conv3d(cout, cout, kernel_size=3, padding=1),
                nn.InstanceNorm3d(cout),
                nn.GELU(),
            ])

        # ADD THE 4TH UPSAMPLE STEP HERE: 32->32
        last_channel = channels[-1]
        layers.extend([
            nn.ConvTranspose3d(last_channel, last_channel, kernel_size=2, stride=2),
            nn.Conv3d(last_channel, last_channel, kernel_size=3, padding=1),
            nn.InstanceNorm3d(last_channel),
            nn.GELU(),
        ])

        self.decoder = nn.Sequential(*layers)
        self.out = nn.Conv3d(last_channel, out_channels, kernel_size=1)

    def forward(self, x):
        return nn.functional.softplus(self.out(self.decoder(x)))
    

import torch
import torch.nn as nn

class LatentRegressor(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, dropout_prob=0.2):
        """
        in_channels: The number of feature channels coming out of your Latent Transformer/Bottleneck.
        hidden_dim: The number of neurons in the hidden layers.
        """
        super().__init__()
        
        # 1. Spatial Pooling
        # This takes [Batch, Channels, D, H, W] and averages the spatial dimensions
        # down to [Batch, Channels, 1, 1, 1]. It makes the MLP invariant to the exact 
        # spatial size of your bottleneck.
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # 2. MLP Regression Head
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm stabilizes regression targets brilliantly
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            
            # Output layer: exactly 9 values
            # (3 for anchor, 3 for end_offset, 3 for bragg_offset)
            nn.Linear(hidden_dim // 2, 9)
        )

    def forward(self, latent_features):
        """
        latent_features: Tensor of shape [Batch, Channels, D, H, W]
        Returns: Tensor of shape [Batch, 9]
        """
        # Pool spatial dimensions
        x = self.pool(latent_features)
        
        # Flatten from [Batch, Channels, 1, 1, 1] to [Batch, Channels]
        x = torch.flatten(x, start_dim=1)
        
        # Predict the 9 physical parameters
        predictions = self.mlp(x)
        
        return predictions


import torch
import torch.nn as nn

class FiLM3D(nn.Module):
    def __init__(self, condition_dim=9, latent_channels=256):
        """
        condition_dim: Number of predicted coordinates (e.g., 9 for start, end, max)
        latent_channels: The number of feature channels in your 3D bottleneck
        """
        super().__init__()
        self.latent_channels = latent_channels
        
        # A small MLP to convert the 9 coordinates into scaling (gamma) and shifting (beta) factors
        # It outputs 2 * latent_channels (one gamma and one beta for every channel)
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_channels * 2)
        )
        
        # Initialize the final layer to output zeros so we start with an identity transform
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, latent_features, predicted_coords):
        """
        latent_features: Tensor of shape [Batch, Channels, D, H, W]
        predicted_coords: Tensor of shape [Batch, condition_dim]
        """
        batch_size = latent_features.size(0)
        
        # Pass coordinates through the MLP
        film_params = self.mlp(predicted_coords)
        
        # Split the output into gamma (scale) and beta (shift)
        # Each will have shape [Batch, Channels]
        delta_gamma, beta = torch.split(film_params, self.latent_channels, dim=1)
        
        # Add 1 to gamma so the default scale is 1 (identity)
        gamma = 1.0 + delta_gamma
        
        # Reshape to [Batch, Channels, 1, 1, 1] so they broadcast over the 3D spatial dimensions
        gamma = gamma.view(batch_size, self.latent_channels, 1, 1, 1)
        beta = beta.view(batch_size, self.latent_channels, 1, 1, 1)
        
        # Apply the modulation
        conditioned_features = (latent_features * gamma) + beta
        
        return conditioned_features


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
    def __init__(self,embed_dim=256,num_heads=8,num_layers=4,mlp_ratio=4,use_fourier_embedding=False,num_freqs=8):
        super().__init__()

        # Beam encoder
        

        self.energy_encoder = nn.Sequential(nn.Linear(1, 64),nn.GELU())
        if use_fourier_embedding:
            self.energy_encoder = FourierFeatureEmbedder(num_freqs=num_freqs)
        cond_input = self.energy_encoder.out_dim if use_fourier_embedding else 64
        self.condition_proj = nn.Sequential(nn.Linear(cond_input,embed_dim),nn.GELU())
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
        energy = self.energy_encoder(beam[:, 2:3])
        condition = self.condition_proj(energy)

        # Add conditioning to every token
        x = x + condition.unsqueeze(1)

        x = self.transformer(x)

        # back to volume
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x

class DoTA_based(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, channels=(32, 64, 128, 256), num_heads=8, num_layers=4,use_fourier_embedding=False,num_freqs=8):
        super().__init__()
        embed_dim = channels[-1]
        self.encoder = Encoder(in_channels=in_channels, channels=channels)
        self.transformer = ConditionedTransformer(embed_dim=embed_dim,num_heads=num_heads,num_layers=num_layers,use_fourier_embedding=use_fourier_embedding,num_freqs=num_freqs)
        self.decoder = Decoder(channels=list(reversed(channels)), out_channels=out_channels)

    def forward(self, x, beam):
        x = self.encoder(x)
        x = self.transformer(x, beam)
        x = self.decoder(x)
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet

from monai.networks.nets import SwinUNETR

class DoseSwinUnet(nn.Module):
    def __init__(self, condition_dim, encoded_cond_features=4):
        super().__init__()
        
        # Your existing MLP condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Linear(64, encoded_cond_features)
        )
        
        # The Swin Transformer Hybrid
        self.unet = SwinUNETR(
            img_size=(128, 128, 32), # MUST match your ResizeWithPadOrCropd spatial_size!
            in_channels=2 + encoded_cond_features,
            out_channels=1,
            feature_size=24, # Base feature size (increase to 48 if you have massive VRAM)
            use_checkpoint=True, # Saves VRAM during training by trading compute for memory
            spatial_dims=3
        )

    def forward(self, ct, condition):
        # Your existing forward pass works perfectly here!
        B, _, X, Y, Z = ct.shape
        flat_cond = self.cond_encoder(condition)
        spatially_aware_cond = flat_cond.view(B, -1, 1, 1, 1)
        broadcast_cond = spatially_aware_cond.expand(B, -1, X, Y, Z)
        
        unet_input = torch.cat([ct, broadcast_cond], dim=1)
        return self.unet(unet_input)
    

from monai.networks.nets import AttentionUnet

class DoseAttentionUnet(nn.Module):
    def __init__(self, condition_dim, encoded_cond_features=4,cond_encoder = None):
        super().__init__()
        
        # Your existing MLP condition encoder
        self.cond_encoder = cond_encoder if cond_encoder is not None else nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Linear(64, encoded_cond_features)
        )
        
        # The Attention-Gated UNet
        self.unet = AttentionUnet(
            spatial_dims=3,
            in_channels=2 + encoded_cond_features,
            out_channels=1,
            # We use the exact same channels and strides as your original UNet
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.1
        )

    def forward(self, ct, condition):
        # Your existing forward pass works perfectly here too!
        B, _, X, Y, Z = ct.shape
        flat_cond = self.cond_encoder(condition)
        spatially_aware_cond = flat_cond.view(B, -1, 1, 1, 1)
        broadcast_cond = spatially_aware_cond.expand(B, -1, X, Y, Z)
        
        unet_input = torch.cat([ct, broadcast_cond], dim=1)
        return self.unet(unet_input)



import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet

class FiLMConditionalDoseUnet(nn.Module):
    def __init__(self, condition_dim, bottleneck_channels=256):
        super().__init__()
        
        # 1. BasicUNet exposes its layers explicitly, allowing safe interception!
        self.unet = BasicUNet(
            spatial_dims=3,
            in_channels=2,  # Only CT goes in. Condition is injected deep inside.
            out_channels=1,
            # BasicUNet requires 6 feature sizes: 5 for encoder/bottleneck, 1 for the final decoder block
            features=(16, 32, 64, 128, bottleneck_channels, 16),
            dropout=0.1
        )
        
        # 2. The FiLM Generator
        # Outputs Scale (Gamma) and Shift (Beta) for the 256 bottleneck channels
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_channels * 2) 
        )

    def forward(self, ct, condition):
        # 1. Generate the Scale (Gamma) and Shift (Beta) factors
        film_params = self.film_generator(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # Reshape for 3D broadcasting
        gamma = gamma.view(ct.shape[0], -1, 1, 1, 1)
        beta = beta.view(ct.shape[0], -1, 1, 1, 1)

        # 2. Encoder (Using MONAI's exact downsampling layer names)
        x0 = self.unet.conv_0(ct)
        x1 = self.unet.down_1(x0)
        x2 = self.unet.down_2(x1)
        x3 = self.unet.down_3(x2)
        
        # 3. The Bottleneck (Deepest feature representation)
        bottleneck = self.unet.down_4(x3)
        
        # --- INJECT THE PHYSICS PRIOR (FiLM) ---
        # The geometric features are mathematically scaled and shifted by the beam angle
        bottleneck = bottleneck * (1 + gamma) + beta
        
        # 4. Decoder (Using MONAI's 'upcat' layers which expect the bottleneck AND the skip connection)
        u4 = self.unet.upcat_4(bottleneck, x3)
        u3 = self.unet.upcat_3(u4, x2)
        u2 = self.unet.upcat_2(u3, x1)
        u1 = self.unet.upcat_1(u2, x0)
        
        # 5. Final output convolution
        out = self.unet.final_conv(u1)
        
        return out
    


class ConditionalDoseUNet(nn.Module):
    def __init__(self, condition_dim=3, encoded_cond_features=8,use_softplus=True):
        """
        Args:
            condition_dim: The length of your raw condition vector (e.g., 3 for [x, y, z])
            encoded_cond_features: How many feature channels the condition will occupy 
                                   when concatenated with the CT.
        """
        super().__init__()
        
        # 1. Condition Encoder (MLP)
        # Transforms the raw physics numbers into rich, network-readable features
        self.cond_encoder = nn.Sequential(
            # First linear expansion
            nn.Linear(condition_dim, 32),
            nn.LayerNorm(32), 
            nn.ReLU(),
            
            # Final compression down to the broadcast channels
            nn.Linear(32, encoded_cond_features)
        )
        
        # 2. The MONAI UNet (Simplified Architecture)
        # We start with 16 channels instead of 32/64 to prevent overfitting the small ROI
        self.unet = UNet(
            spatial_dims=3,
            # CT Channel (1) + Prior Mask Channel (1) + Encoded Condition Channels
            in_channels=2 + encoded_cond_features,
            out_channels=1,
            channels=(16, 32, 64, 128, 256), # 5 layers deep
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
            dropout=0.1, # Light dropout to further regularize
        )
        self.use_softplus = use_softplus

    def forward(self, ct_volume, condition_vector):
        """
        ct_volume: (Batch, 1, 64, 64, 32)
        condition_vector: (Batch, condition_dim)
        """
        B, _, X, Y, Z = ct_volume.shape
        
        # Step 1: Encode the condition vector
        # Shape goes from (B, cond_dim) -> (B, encoded_cond_features)
        cond_feat = self.cond_encoder(condition_vector)
        
        # Step 2: Spatial Broadcasting
        # Reshape to (B, Features, 1, 1, 1) and expand to match the CT dimensions
        cond_feat_spatial = cond_feat.view(B, -1, 1, 1, 1).expand(-1, -1, X, Y, Z)
        
        # Step 3: Concatenate along the channel dimension (dim=1)
        # Resulting shape: (B, 1 + encoded_cond_features, 64, 64, 32)
        x = torch.cat([ct_volume, cond_feat_spatial], dim=1)
        
        # Step 4: UNet Forward Pass
        raw_output = self.unet(x)
        
        # Step 5: The Softplus Fix
        # Prevents the Dying ReLU problem in the empty background
        dose_pred = F.softplus(raw_output) if self.use_softplus else F.relu(raw_output)
        
        return dose_pred