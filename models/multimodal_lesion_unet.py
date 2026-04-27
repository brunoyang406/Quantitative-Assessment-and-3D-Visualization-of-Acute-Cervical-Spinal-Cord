"""
Multimodal lesion U-Net (`multimodal_lesion_unet`): one encoder per modality, dual CBAM fusion
per stage, shared decoder, optional Swin blocks, multi-scale deep supervision (logits at native
decoder resolutions; no in-model upsampling for aux heads).

Backward-compatible alias: `MultiModalUNetWithDualCBAM` (end of file).
"""

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit
from .cbam import CBAM3D
from .swin_transformer_block import SwinTransformerBlock3D


class EncoderOnly(nn.Module):
    """Single-modality encoder: one block per stage, returns per-stage feature list."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        channels: tuple = (32, 64, 128, 256, 320, 320, 320),
        strides: tuple = ((1,1,1), (1,2,2), (1,2,2), (2,2,2), (1,2,2), (1,2,2), (1,2,2)),
        kernel_sizes: tuple = None,
        num_res_units: int = 2,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.strides = strides
        
        if kernel_sizes is None:
            kernel_sizes = [
                (1, 3, 3) if i < 2 else (3, 3, 3) for i in range(len(strides))
            ]
        self.kernel_sizes = kernel_sizes
        
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        encoder_channels = []
        encoder_channel_list = channels[:len(strides)]
        for out_ch, stride, kernel_size in zip(encoder_channel_list, strides, kernel_sizes):
            encoder_block = self._make_encoder_block(
                in_ch, out_ch, stride, kernel_size, num_res_units, norm, dropout
            )
            self.encoder.append(encoder_block)
            encoder_channels.append(out_ch)
            in_ch = out_ch
        
        self.encoder_channels = encoder_channels
    
    def _make_encoder_block(self, in_ch, out_ch, stride, kernel_size, num_res_units, norm, dropout):
        layers = []
        
        layers.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=in_ch,
                out_channels=out_ch,
                strides=stride,
                kernel_size=kernel_size,
                norm=norm,
                act="relu",
                dropout=dropout,
            )
        )
        
        for _ in range(num_res_units):
            layers.append(
                ResidualUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=out_ch,
                    out_channels=out_ch,
                    strides=1,
                    kernel_size=kernel_size,
                    subunits=2,
                    adn_ordering="NDA",
                    act="relu",
                    norm=norm,
                    dropout=dropout,
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            features.append(x)
        return features


class MultiModalStageFusion(nn.Module):
    """Per-stage fusion: CBAM per modality → concat → 1×1 conv → CBAM on fused tensor."""
    
    def __init__(
        self,
        channels: int,
        spatial_dims: int = 3,
        cbam_reduction: int = 16,
        dropout: float = 0.0,
        include_cord: bool = True,
        include_uncertainty: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.spatial_dims = spatial_dims
        self.include_cord = include_cord
        self.include_uncertainty = include_uncertainty
        
        self.t1_cbam = CBAM3D(gate_channels=channels, reduction_ratio=cbam_reduction, no_spatial=False)
        self.t2_cbam = CBAM3D(gate_channels=channels, reduction_ratio=cbam_reduction, no_spatial=False)
        self.t2fs_cbam = CBAM3D(gate_channels=channels, reduction_ratio=cbam_reduction, no_spatial=False)
        
        num_modalities = 3
        if include_cord:
            self.cord_cbam = CBAM3D(gate_channels=channels, reduction_ratio=cbam_reduction, no_spatial=False)
            num_modalities += 1
        if include_uncertainty:
            self.uncertainty_cbam = CBAM3D(gate_channels=channels, reduction_ratio=cbam_reduction, no_spatial=False)
            num_modalities += 1
        
        self.fusion = nn.Conv3d(channels * num_modalities, channels, kernel_size=1)
        
        if dropout > 0.0:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = None
        
        self.fused_cbam = CBAM3D(gate_channels=channels, reduction_ratio=cbam_reduction, no_spatial=False)
    
    def forward(
        self, 
        t1_feat: torch.Tensor, 
        t2_feat: torch.Tensor, 
        t2fs_feat: torch.Tensor,
        cord_feat: torch.Tensor = None,
        uncertainty_feat: torch.Tensor = None
    ):
        t1_attended = self.t1_cbam(t1_feat)
        t2_attended = self.t2_cbam(t2_feat)
        t2fs_attended = self.t2fs_cbam(t2fs_feat)
        
        features_to_concat = [t1_attended, t2_attended, t2fs_attended]
        
        if self.include_cord and cord_feat is not None:
            cord_attended = self.cord_cbam(cord_feat)
            features_to_concat.append(cord_attended)
        
        if self.include_uncertainty and uncertainty_feat is not None:
            uncertainty_attended = self.uncertainty_cbam(uncertainty_feat)
            features_to_concat.append(uncertainty_attended)
        
        concatenated = torch.cat(features_to_concat, dim=1)
        
        fused = self.fusion(concatenated)
        
        if self.dropout is not None:
            fused = self.dropout(fused)
        
        fused_attended = self.fused_cbam(fused)
        
        return fused_attended


class DecoderOnly(nn.Module):
    """UNet decoder: transposed conv upsample → concat skip → conv block; optional list return for deep supervision."""

    def __init__(
        self,
        spatial_dims: int = 3,
        channels: tuple = (32, 64, 128, 256, 320, 320, 320),
        strides: tuple = ((1,1,1), (1,2,2), (1,2,2), (2,2,2), (1,2,2), (1,2,2), (1,2,2)),
        num_res_units: int = 2,
        norm: str = "instance",
        dropout: float = 0.0,
        bottleneck_ch: int = None,
        deep_supervision: bool = True,
        deep_supervision_heads: int = 3,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.strides = strides
        self.num_res_units = num_res_units
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        start_ch = bottleneck_ch if bottleneck_ch is not None else channels[-1]
        prev_decoder_out_ch = start_ch
        
        reversed_strides = list(strides[::-1])
        
        for i in range(len(channels)):
            stride = reversed_strides[i]
            skip_ch = channels[len(channels) - 1 - i]
            
            upsample_layer = self._make_upsample_layer(prev_decoder_out_ch, skip_ch, stride)
            self.upsample_layers.append(upsample_layer)
            
            conv_block = self._make_conv_block(
                in_channels=skip_ch * 2,
                out_channels=skip_ch,
                norm=norm,
                dropout=dropout,
            )
            self.conv_blocks.append(conv_block)
            
            prev_decoder_out_ch = skip_ch
        
        self.final_channels = prev_decoder_out_ch
    
    def _make_upsample_layer(self, in_channels, out_channels, stride):
        """Transposed conv; forward() may interpolate to match skip spatial size."""
        kernel_size = tuple(s for s in stride)
        stride_tuple = tuple(stride)
        
        return nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_tuple,
            padding=0,
        )
    
    def _make_conv_block(self, in_channels, out_channels, norm, dropout):
        layers = []
        
        layers.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=1,
                norm=norm,
                act="relu",
                dropout=dropout,
                conv_only=False,
            )
        )
        
        for _ in range(self.num_res_units):
            layers.append(
                ResidualUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    subunits=2,
                    adn_ordering="NDA",
                    act="relu",
                    norm=norm,
                    dropout=dropout,
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, bottleneck_feat: torch.Tensor, skip_features: list):
        x = bottleneck_feat
        n = len(self.channels)
        
        decoder_outputs = []
        
        for i, (upsample_layer, conv_block) in enumerate(zip(self.upsample_layers, self.conv_blocks)):
            skip_idx = n - 1 - i
            skip = skip_features[skip_idx]
            target_shape = skip.shape[2:]
            
            x = upsample_layer(x)
            
            if x.shape[2:] != target_shape:
                x = torch.nn.functional.interpolate(
                    x, size=target_shape, mode='trilinear', align_corners=False
                )
            
            x = torch.cat([x, skip], dim=1)
            
            x = conv_block(x)
            
            if self.deep_supervision:
                decoder_outputs.append(x)
        
        if self.deep_supervision:
            return decoder_outputs
        else:
            return x


class SwinBottleneck(nn.Module):
    """Two Swin blocks: window attention, then shifted-window attention."""

    def __init__(self, dim, resolution, num_heads=10, window_size=(4, 4, 2), mlp_ratio=4., dropout=0.):
        super().__init__()
        self.resolution = resolution
        self.swin1 = SwinTransformerBlock3D(
            dim=dim, input_resolution=resolution, num_heads=num_heads, 
            window_size=window_size, shift_size=(0, 0, 0), mlp_ratio=mlp_ratio, drop=dropout
        )
        shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.swin2 = SwinTransformerBlock3D(
            dim=dim, input_resolution=resolution, num_heads=num_heads, 
            window_size=window_size, shift_size=shift_size, mlp_ratio=mlp_ratio, drop=dropout
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        x = self.swin1(x)
        x = self.swin2(x)
        
        x = x.transpose(1, 2).view(B, C, D, H, W)
        return x


class MultimodalLesionUNet(nn.Module):
    """See module docstring; `deep_supervision_heads` controls how many aux logits are returned."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        out_channels: int = 256,
        channels: tuple = (32, 64, 128, 256, 320, 320, 320, 320),
        strides: tuple = ((1,1,1), (1,2,2), (1,2,2), (2,2,2), (1,2,2), (1,2,2), (1,2,2)),
        kernel_sizes: tuple = None,
        num_res_units: int = 2,
        norm: str = "instance",
        dropout: float = 0.0,
        cbam_reduction: int = 16,
        deep_supervision: bool = True,
        deep_supervision_heads: int = 3,
        include_cord: bool = True,
        include_uncertainty: bool = False,
        use_swin_bottleneck: bool = True,
        swin_from_stage: int = 4,
        input_size: tuple = (16, 512, 256),
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        self.include_cord = include_cord
        self.include_uncertainty = include_uncertainty
        self.use_swin_bottleneck = use_swin_bottleneck
        self.swin_from_stage = swin_from_stage
        
        self.stage_resolutions = []
        curr_res = list(input_size)
        for s in strides:
            curr_res = [curr_res[i] // s[i] for i in range(3)]
            self.stage_resolutions.append(tuple(curr_res))
            
        self.bottleneck_res = self.stage_resolutions[-1]
        
        encoder_channel_list = channels[:len(strides)]
        bottleneck_ch = encoder_channel_list[-1]
        
        self.t1_encoder = EncoderOnly(
            spatial_dims=spatial_dims,
            in_channels=1,
            channels=encoder_channel_list,
            strides=strides,
            kernel_sizes=kernel_sizes,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        )
        self.t2_encoder = EncoderOnly(
            spatial_dims=spatial_dims,
            in_channels=1,
            channels=encoder_channel_list,
            strides=strides,
            kernel_sizes=kernel_sizes,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        )
        self.t2fs_encoder = EncoderOnly(
            spatial_dims=spatial_dims,
            in_channels=1,
            channels=encoder_channel_list,
            strides=strides,
            kernel_sizes=kernel_sizes,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        )
        
        if include_cord:
            self.cord_encoder = EncoderOnly(
                spatial_dims=spatial_dims,
                in_channels=1,
                channels=encoder_channel_list,
                strides=strides,
                kernel_sizes=kernel_sizes,
                num_res_units=num_res_units,
                norm=norm,
                dropout=dropout,
            )
        
        if include_uncertainty:
            self.uncertainty_encoder = EncoderOnly(
                spatial_dims=spatial_dims,
                in_channels=1,
                channels=encoder_channel_list,
                strides=strides,
                kernel_sizes=kernel_sizes,
                num_res_units=num_res_units,
                norm=norm,
                dropout=dropout,
            )
        
        self.stage_fusion_modules = nn.ModuleList([
            MultiModalStageFusion(
                channels=ch,
                spatial_dims=spatial_dims,
                cbam_reduction=cbam_reduction,
                dropout=dropout,
                include_cord=include_cord,
                include_uncertainty=include_uncertainty,
            )
            for ch in encoder_channel_list
        ])
        
        self.bottleneck_fusion = MultiModalStageFusion(
            channels=bottleneck_ch,
            spatial_dims=spatial_dims,
            cbam_reduction=cbam_reduction,
            dropout=dropout,
            include_cord=include_cord,
            include_uncertainty=include_uncertainty,
        )
        
        self.bottleneck = self._make_bottleneck_block(
            bottleneck_ch, bottleneck_ch, num_res_units, norm, dropout, resolution=self.bottleneck_res
        )
        
        self.decoder = DecoderOnly(
            spatial_dims=spatial_dims,
            channels=tuple(encoder_channel_list),
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
            bottleneck_ch=bottleneck_ch,
            deep_supervision=deep_supervision,
            deep_supervision_heads=deep_supervision_heads,
        )
        
        self.final_layer = nn.Conv3d(
            encoder_channel_list[0], out_channels, kernel_size=1
        )
        
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            
            n_decoder_stages = len(encoder_channel_list)
            for i in range(min(deep_supervision_heads, n_decoder_stages)):
                decoder_ch = encoder_channel_list[i]
                
                ds_head = nn.Conv3d(decoder_ch, out_channels, kernel_size=1)
                self.ds_heads.append(ds_head)
    
    def _make_bottleneck_block(self, in_ch, out_ch, num_res_units, norm, dropout, resolution=None):
        layers = []
        
        if self.use_swin_bottleneck and resolution is not None:
            window_size = (resolution[0] // 2, resolution[1] // 2, resolution[2] // 2)
            window_size = tuple(max(2, w) for w in window_size)
            
            layers.append(
                SwinBottleneck(
                    dim=in_ch, 
                    resolution=resolution, 
                    num_heads=in_ch // 32, 
                    window_size=window_size,
                    dropout=dropout
                )
            )
        
        for _ in range(max(1, num_res_units // 2)):
            layers.append(
                ResidualUnit(
                    spatial_dims=self.spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    strides=1,
                    kernel_size=3,
                    subunits=2,
                    adn_ordering="NDA",
                    act="relu",
                    norm=norm,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        t1 = x[:, 0:1]
        t2 = x[:, 1:2]
        t2fs = x[:, 2:3]
        
        t1_features = self.t1_encoder(t1)
        t2_features = self.t2_encoder(t2)
        t2fs_features = self.t2fs_encoder(t2fs)
        
        channel_idx = 3
        if self.include_cord:
            cord = x[:, channel_idx:channel_idx+1]
            cord_features = self.cord_encoder(cord)
            channel_idx += 1
        else:
            cord_features = None
        
        if self.include_uncertainty:
            uncertainty = x[:, channel_idx:channel_idx+1]
            uncertainty_features = self.uncertainty_encoder(uncertainty)
            channel_idx += 1
        else:
            uncertainty_features = None
        
        fused_features = []
        for i, (t1_f, t2_f, t2fs_f, fusion_module) in enumerate(
            zip(t1_features, t2_features, t2fs_features, self.stage_fusion_modules)
        ):
            cord_f = cord_features[i] if self.include_cord and cord_features is not None else None
            uncertainty_f = uncertainty_features[i] if self.include_uncertainty and uncertainty_features is not None else None
            fused = fusion_module(t1_f, t2_f, t2fs_f, cord_f, uncertainty_f)
            fused_features.append(fused)
        
        bottleneck_input = self.bottleneck_fusion(
            t1_features[-1], 
            t2_features[-1], 
            t2fs_features[-1],
            cord_features[-1] if self.include_cord and cord_features is not None else None,
            uncertainty_features[-1] if self.include_uncertainty and uncertainty_features is not None else None
        )
        
        del t1_features, t2_features, t2fs_features, cord_features, uncertainty_features
        
        bottleneck_output = self.bottleneck(bottleneck_input)
        
        decoded = self.decoder(bottleneck_output, fused_features)
        
        if not self.deep_supervision:
            output = self.final_layer(decoded)
            return output
        
        decoder_features = decoded
        
        main_feature = decoder_features[-1]
        main_output = self.final_layer(main_feature)
        
        ds_outputs = []
        for i in range(min(self.deep_supervision_heads, len(decoder_features))):
            decoder_feat = decoder_features[-(i+1)]
            ds_logits = self.ds_heads[i](decoder_feat)
            ds_outputs.append(ds_logits)
        
        return [main_output] + ds_outputs


# Historical name kept for backward compatibility with external imports and notes.
MultiModalUNetWithDualCBAM = MultimodalLesionUNet

