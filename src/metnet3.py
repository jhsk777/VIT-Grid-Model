import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from typing import List
from collections import OrderedDict
from classification import categorical_to_continuous
from maxvit import MaxViT
import numpy as np
import ipdb


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, default_mean=None, default_std=None):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
        self.default_mean = default_mean
        self.default_std = default_std

    def forward(self, x, mode:str):
        if mode == 'norm':
            #self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'denorm2':
            x = self._denormalize2(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features, dtype=torch.float32))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features, dtype=torch.float32))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        mask = ~torch.isnan(x)
        valid_counts = mask.sum(dim=dim2reduce, keepdim=True)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        sum_squared_diff = torch.where(mask, (x - self.mean)**2, torch.zeros_like(x)).sum(dim=dim2reduce, keepdim=True)
        variance = sum_squared_diff / valid_counts
        
        self.stdev = torch.sqrt(variance + self.eps).detach()
        self.mean = torch.where(torch.isnan(self.mean), torch.FloatTensor([self.default_mean]).cuda(), self.mean)
        self.stdev = torch.where(torch.isnan(self.stdev), torch.FloatTensor([self.default_std]).cuda(), self.stdev)
        self.stdev = torch.where(self.stdev==0, torch.FloatTensor([self.default_std]).cuda(), self.stdev)
        # self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def _denormalize2(self, x):
        if self.affine:
            x = x - self.affine_bias[:x.shape[2]]
            x = x / (self.affine_weight + self.eps*self.eps)[:x.shape[2]]
        
        x = x * self.stdev[:,:,:x.shape[2]]
        x = x + self.mean[:,:,:x.shape[2]]
        return x


Downsample2x = partial(nn.MaxPool2d, kernel_size = 2, stride = 2)

def Upsample2x(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, kernel_size = 2, stride = 2)


# they use layernorms after the conv in the resnet blocks for some reason

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = self.eps).rsqrt() * self.g + self.b

# Resnet module

# conditionable resnet block

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = ChanLayerNorm(dim_out)
        self.act = nn.ReLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        cond_dim = None,
    ):
        super().__init__()
        self.mlp = None

        if cond_dim is not None:
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cond_dim, dim_out * 2)
            )

        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, cond = None):

        scale_shift = None

        if self.mlp is not None and cond is not None:
            cond = self.mlp(cond)
            cond = rearrange(cond, 'b c -> b c 1 1')
            scale_shift = cond.chunk(2, dim = 1)
        
        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
class ResnetBlocks(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        depth = 1,
        cond_dim = None
    ):
        super().__init__()
        curr_dim = dim_in

        blocks = []
        for _ in range(depth):
            blocks.append(ResnetBlock(dim_in = curr_dim, dim_out = dim_out, cond_dim = cond_dim))
            curr_dim = dim_out

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond = None):

        for block in self.blocks:
            x = block(x, cond = cond)

        return x


    
class MetNet3(nn.Module):
    def __init__(
        self,
        input_size_sample: tuple,  # window_size, n_variables, height, width
        n_start_channels: int,
        end_lead_time: int,
        pm25_boundaries: List[float],
        pm10_boundaries: List[float],
        pm25_mean: float,
        pm25_std: float,
        lead_time_emb_dim: int = 2,
        model_time_emb_dim: int = 1,
        concat_time_to_input: bool = True,
        pm25: bool = True,
        pm10: bool = False,
        resnet_block_depth: int = 2,
        direct_regional: bool = False,
        ignore_backbone: bool = False,
        # MaxVit related settings
        vit_block_depth: int = 1,
        n_heads: int = 32,
        dim_head: int = 32,
        vit_window_size: int = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        num_register_tokens: int = 4,
        normalization_method = "Standard",
    ):
        super(MetNet3, self).__init__()

        window_size, n_variables, input_height, input_width = input_size_sample
        self.window_size = window_size
        self.n_variables = n_variables
        self.input_height = input_height
        self.input_width = input_width
        self.n_input_channels = window_size*n_variables
        self.n_start_channels = n_start_channels
        self.end_lead_time = end_lead_time
        self.concat_time_to_input = concat_time_to_input
        self.vit_window_size = vit_window_size
        self.pm25_mean = pm25_mean
        self.pm25_std = pm25_std
        self.normalization_method = normalization_method
        
        if not direct_regional:
            assert ignore_backbone == False
        
        self.direct_regional = direct_regional
        self.ignore_backbone = ignore_backbone

        self.pm25 = pm25
        self.pm10 = pm10
        if not pm25 and not pm10:
            raise ValueError("At least one of pm_2_5 and pm_10 must be True")
        if pm25:
            if pm25_boundaries is None:
                raise ValueError("pm25_boundaries must be provided")
            self.register_buffer('pm25_boundaries', torch.FloatTensor(pm25_boundaries))
        if pm10:
            if pm10_boundaries is None:
                raise ValueError("pm10_boundaries must be provided")
            self.register_buffer('pm10_boundaries', torch.FloatTensor(pm10_boundaries))
        
        if self.normalization_method == "RevIN":
            self.revin_layer = RevIN(self.input_height * self.input_width, default_mean=self.pm25_mean, default_std=self.pm25_std)
        # lead time embedding
        self.condition_lead_time = nn.Embedding(
            end_lead_time + 1,
            lead_time_emb_dim
        )

        # model time embedding
        self.condition_model_time = nn.ModuleList([
            nn.Embedding(12 + 1, model_time_emb_dim),  # month embedding
            nn.Embedding(31 + 1, model_time_emb_dim),  # day embedding
            nn.Embedding(24 + 1, model_time_emb_dim),  # model hour embedding
        ])

        self.resnet1 = ResnetBlocks(
            dim_in = self.n_input_channels + lead_time_emb_dim + model_time_emb_dim*3 if self.concat_time_to_input
                else self.n_input_channels,
            dim_out = n_start_channels,
            cond_dim = lead_time_emb_dim,
            depth = resnet_block_depth
        )

        self.down = Downsample2x()


        self.vit = MaxViT(
            dim = n_start_channels,
            depth = vit_block_depth,
            cond_dim = lead_time_emb_dim,
            heads = n_heads,
            dim_head = dim_head,
            vit_window_size = vit_window_size,
            mbconv_expansion_rate = mbconv_expansion_rate,
            mbconv_shrinkage_rate = mbconv_shrinkage_rate,
            dropout = dropout,
            num_register_tokens = num_register_tokens
        )

        self.up = Upsample2x(n_start_channels, n_start_channels)

        self.resnet2 = ResnetBlocks(
            dim_in = n_start_channels,
            dim_out = n_start_channels,
            cond_dim = lead_time_emb_dim,
            depth = resnet_block_depth
        )


        # Create out convolution block
        if pm25:
            self.classifier_pm25 = nn.Conv2d(n_start_channels, 1, kernel_size=1)
            if direct_regional:
                self.regr_regional_pm25 = nn.Sequential(
                    nn.Conv2d(n_start_channels, 1, kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(input_height * input_width, 19)
                )

        if pm10:
            self.classifier_pm10 = nn.Conv2d(n_start_channels, len(pm10_boundaries) + 1, kernel_size=1)
            if direct_regional:
                self.regr_regional_pm10 = nn.Sequential(
                    nn.Conv2d(n_start_channels, 1, kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(input_height * input_width, 19)
                )

    
    def pad(self, data, constant=(0, 0), pad_size=14):
        H, W = data.shape[-2:]
        pad_h = (pad_size - H) % pad_size
        pad_w = (pad_size - W) % pad_size

        pad_values = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

        data = F.pad(data, pad_values, value=constant[0])

        return data, pad_values

    def unpad(self, data, pad_values):
        pad_left, pad_right, pad_top, pad_bottom = pad_values
        return data[..., pad_top:-pad_bottom, pad_left:-pad_right]

    def forward(self, x, labels_pm25=None, region_targets_pm25=None, labels_pm10=None, region_targets_pm10=None, timestamps: torch.Tensor = None, prev_vals: torch.Tensor = None):
        """
        x: (B, T, C, H, W)  # NWP outputs (T=window_size, C=n_variables)
        labels_pm25: (B, H, W)  # ground truth PM2.5 values
        region_targets_pm25: (B, R=19)  # ground truth PM2.5 values for each region
        labels_pm10: (B, H, W)  # ground truth PM10 values
        region_targets_pm10: (B, R=19)  # ground truth PM10 values for each region
        timestamps: (B, 5)  # (year, month, day, hour, lead_time)
        """
        # if self.pm25 and (labels_pm25 is None or region_targets_pm25 is None):
        #     raise ValueError("labels_pm25 and region_targets_pm25 must be provided")
        # if self.pm10 and (labels_pm10 is None or region_targets_pm10 is None):
        #     raise ValueError("labels_pm10 and region_targets_pm10 must be provided")

        
        lead_times_to_predict = torch.arange(1, self.end_lead_time + 1, device=x.device)
        B = x.shape[0]
        norm_pm_0 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()
        norm_pm_1 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()
        norm_pm_2 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()
        norm_pm_3 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()

        for i in range(self.window_size):
            cur_indices = np.array([4, 10, 16, 22])
            cur_sim_vals = x[:,i, cur_indices, :,:]
            norm_pm_0[:, i] = cur_sim_vals[:,0,:,:].reshape(B, self.input_height * self.input_width)
            norm_pm_1[:, i] = cur_sim_vals[:,1,:,:].reshape(B, self.input_height * self.input_width)
            norm_pm_2[:, i] = cur_sim_vals[:,2,:,:].reshape(B, self.input_height * self.input_width)
            norm_pm_3[:, i] = cur_sim_vals[:,3,:,:].reshape(B, self.input_height * self.input_width)
        
        if self.normalization_method == "Standard":
            norm_pm_0 = (norm_pm_0 - self.pm25_mean) / self.pm25_std
            norm_pm_1 = (norm_pm_1 - self.pm25_mean) / self.pm25_std
            norm_pm_2 = (norm_pm_2 - self.pm25_mean) / self.pm25_std
            norm_pm_3 = (norm_pm_3 - self.pm25_mean) / self.pm25_std
    
        x = x.clone()
        for i in range(self.window_size):
            x[:, i, 4, :, :] = norm_pm_0[:, i].reshape(B, self.input_height, self.input_width)
            x[:, i, 10, :, :] = norm_pm_1[:, i].reshape(B, self.input_height, self.input_width)
            x[:, i, 16, :, :] = norm_pm_2[:, i].reshape(B, self.input_height, self.input_width)
            x[:, i, 22, :, :] = norm_pm_3[:, i].reshape(B, self.input_height, self.input_width)
        
        L = len(lead_times_to_predict)
        x = x.repeat_interleave(L, dim=0)
        x, pad_values = self.pad(x)
        
        _, T, C, H, W = x.shape
        x = x.view(B*L, -1, H, W)
        
        def condition_time(self, target_time):
            assert target_time.shape[-1] == 5
            lead_time = target_time[:, -1].to(x.device).int()
            model_time = target_time[:, 1:-1].to(x.device).int()

            lead_time_emb = self.condition_lead_time(lead_time)
            model_time_emb = torch.cat([
                emb(model_time[:, i])
                for i, emb in enumerate(self.condition_model_time)
            ])

            lead_time_emb = lead_time_emb.view(B*L, -1, 1, 1).repeat(1, 1, H, W)
            model_time_emb = model_time_emb.view(B*L, -1, 1, 1).repeat(1, 1, H, W)
            return torch.cat([lead_time_emb, model_time_emb], dim=1)

        # ipdb.set_trace()
        timestamps = timestamps[:,6,:].repeat_interleave(L, dim=0)
        cond = None
        lead_times = lead_times_to_predict.to(x.device).repeat(B) #.view(-1,1)
        cond = self.condition_lead_time(lead_times)
        timestamps = torch.cat([timestamps,lead_times.unsqueeze(-1)],dim=-1)
        # if timestamps is not None:
        #     lead_times = timestamps[:, -1].to(x.device)
        #     cond = self.condition_lead_time(lead_times) # 

        if self.concat_time_to_input:
            time_emb = condition_time(self, timestamps)
            x = torch.cat([x, time_emb], dim=1)
        
        out = self.resnet1(x, cond)
        out = self.down(out)
        out = self.vit(out, cond)
        out = self.up(out)
        out = self.resnet2(out, cond)

        out = self.unpad(out, pad_values) # (B, 128, 82, 67)

        pm_preds = self.classifier_pm25(out)
        pm_preds = pm_preds.squeeze(dim=1).reshape(B,L,self.input_height,self.input_width)
        if self.normalization_method == "Standard":
            pm_preds = pm_preds * self.pm25_std + self.pm25_mean
        return pm_preds

        # ret = OrderedDict()
        # ret["loss"] = 0
        # loss_pm25 = 0
        # loss_pm10 = 0
        # if self.pm25:
        #     nan_mask = torch.isnan(labels_pm25)
        #     labels_pm25 = torch.bucketize(labels_pm25.contiguous(), self.pm25_boundaries, right=True)
        #     labels_pm25[nan_mask] = -100
        #     logits_pm25 = self.classifier_pm25(out)
        #     loss_pm25 = F.cross_entropy(logits_pm25, labels_pm25)
        #     predicted_classes_pm25 = torch.argmax(logits_pm25, dim=1)
        #     predicted_values_pm25 = categorical_to_continuous(predicted_classes_pm25, self.pm25_boundaries)
        #     ret["logits_pm25"] = logits_pm25
        #     ret["predicted_pm25"] = predicted_values_pm25
        #     ret["loss_pm25"] = loss_pm25

        #     if self.direct_regional:
        #         if self.ignore_backbone:
        #             region_preds_pm25 = self.regr_regional_pm25(out.detach())
        #         else:
        #             region_preds_pm25 = self.regr_regional_pm25(out)
        #         region_mask = ~torch.isnan(region_targets_pm25)
        #         regr_loss_pm25 = F.mse_loss(
        #             region_targets_pm25[region_mask], 
        #             region_preds_pm25[region_mask]
        #         )
        #         ret["region_preds_pm25"] = region_preds_pm25
        #         ret["regr_loss_pm25"] = regr_loss_pm25

        # if self.pm10:
        #     nan_mask = torch.isnan(labels_pm10)
        #     labels_pm10 = torch.bucketize(labels_pm10.contiguous(), self.pm10_boundaries, right=True)
        #     labels_pm10[nan_mask] = -100
        #     logits_pm10 = self.classifier_pm10(out)
        #     loss_pm10 = F.cross_entropy(logits_pm10, labels_pm10)
        #     predicted_classes_pm10 = torch.argmax(logits_pm10, dim=1)
        #     predicted_values_pm10 = categorical_to_continuous(predicted_classes_pm10, self.pm10_boundaries)
        #     ret["logits_pm10"] = logits_pm10
        #     ret["predicted_pm10"] = predicted_values_pm10
        #     ret["loss_pm10"] = loss_pm10

        #     if self.direct_regional:
        #         if self.ignore_backbone:
        #             region_preds_pm10 = self.regr_regional_pm10(out.detach())
        #         else:
        #             region_preds_pm10 = self.regr_regional_pm10(out)
        #         region_mask = ~torch.isnan(region_targets_pm10)
        #         regr_loss_pm10 = F.mse_loss(
        #             region_targets_pm10[region_mask], 
        #             region_preds_pm10[region_mask]
        #         )
        #         ret["region_preds_pm10"] = region_preds_pm10
        #         ret["regr_loss_pm10"] = regr_loss_pm10

        # if self.direct_regional:
        #     ret["loss"] = loss_pm25 + loss_pm10 + regr_loss_pm25 + regr_loss_pm10
        # else:
        #    ret["loss"] = loss_pm25 + loss_pm10
        # return ret
    
    def get_ignore_keys_for_eval(self):
        keys = []
        if self.pm25:
            keys += ["loss_pm25", "logits_pm25"]
 
            if self.direct_regional:
                keys += ["regr_loss_pm25"]

        if self.pm10:
            keys += ["loss_pm10", "logits_pm10"]

            if self.direct_regional:
                keys += ["regr_loss_pm10"]
        return keys












class MetNet3_with_stn_imgs(nn.Module):
    def __init__(
        self,
        input_size_sample: tuple,  # window_size, n_variables, height, width
        n_start_channels: int,
        end_lead_time: int,
        pm25_boundaries: List[float],
        pm10_boundaries: List[float],
        pm25_mean: float,
        pm25_std: float,
        lead_time_emb_dim: int = 2,
        model_time_emb_dim: int = 1,
        concat_time_to_input: bool = True,
        pm25: bool = True,
        pm10: bool = False,
        resnet_block_depth: int = 2,
        direct_regional: bool = False,
        ignore_backbone: bool = False,
        # MaxVit related settings
        vit_block_depth: int = 1,
        n_heads: int = 32,
        dim_head: int = 32,
        vit_window_size: int = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        num_register_tokens: int = 4,
        normalization_method = "Standard",
    ):
        super(MetNet3_with_stn_imgs, self).__init__()

        window_size, n_variables, input_height, input_width = input_size_sample
        self.window_size = window_size
        self.n_variables = n_variables
        self.input_height = input_height
        self.input_width = input_width
        self.n_input_channels = window_size*n_variables
        self.n_start_channels = n_start_channels
        self.end_lead_time = end_lead_time
        self.concat_time_to_input = concat_time_to_input
        self.vit_window_size = vit_window_size
        self.pm25_mean = pm25_mean
        self.pm25_std = pm25_std
        self.normalization_method = normalization_method
        
        if not direct_regional:
            assert ignore_backbone == False
        
        self.direct_regional = direct_regional
        self.ignore_backbone = ignore_backbone

        self.pm25 = pm25
        self.pm10 = pm10
        if not pm25 and not pm10:
            raise ValueError("At least one of pm_2_5 and pm_10 must be True")
        if pm25:
            if pm25_boundaries is None:
                raise ValueError("pm25_boundaries must be provided")
            self.register_buffer('pm25_boundaries', torch.FloatTensor(pm25_boundaries))
        if pm10:
            if pm10_boundaries is None:
                raise ValueError("pm10_boundaries must be provided")
            self.register_buffer('pm10_boundaries', torch.FloatTensor(pm10_boundaries))
        
        if self.normalization_method == "RevIN":
            self.revin_layer = RevIN(self.input_height * self.input_width, default_mean=self.pm25_mean, default_std=self.pm25_std)
        # lead time embedding
        self.condition_lead_time = nn.Embedding(
            end_lead_time + 1,
            lead_time_emb_dim
        )

        # model time embedding
        self.condition_model_time = nn.ModuleList([
            nn.Embedding(12 + 1, model_time_emb_dim),  # month embedding
            nn.Embedding(31 + 1, model_time_emb_dim),  # day embedding
            nn.Embedding(24 + 1, model_time_emb_dim),  # model hour embedding
        ])

        self.resnet1 = ResnetBlocks(
            dim_in = self.n_input_channels + lead_time_emb_dim + model_time_emb_dim*3 if self.concat_time_to_input
                else self.n_input_channels,
            dim_out = n_start_channels,
            cond_dim = lead_time_emb_dim,
            depth = resnet_block_depth
        )

        self.down = Downsample2x()


        self.vit = MaxViT(
            dim = n_start_channels,
            depth = vit_block_depth,
            cond_dim = lead_time_emb_dim,
            heads = n_heads,
            dim_head = dim_head,
            vit_window_size = vit_window_size,
            mbconv_expansion_rate = mbconv_expansion_rate,
            mbconv_shrinkage_rate = mbconv_shrinkage_rate,
            dropout = dropout,
            num_register_tokens = num_register_tokens
        )

        self.up = Upsample2x(n_start_channels, n_start_channels)

        self.resnet2 = ResnetBlocks(
            dim_in = n_start_channels,
            dim_out = n_start_channels,
            cond_dim = lead_time_emb_dim,
            depth = resnet_block_depth
        )


        # Create out convolution block
        if pm25:
            self.classifier_pm25 = nn.Conv2d(n_start_channels, 1, kernel_size=1)
            if direct_regional:
                self.regr_regional_pm25 = nn.Sequential(
                    nn.Conv2d(n_start_channels, 1, kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(input_height * input_width, 19)
                )

        if pm10:
            self.classifier_pm10 = nn.Conv2d(n_start_channels, len(pm10_boundaries) + 1, kernel_size=1)
            if direct_regional:
                self.regr_regional_pm10 = nn.Sequential(
                    nn.Conv2d(n_start_channels, 1, kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(input_height * input_width, 19)
                )

    
    def pad(self, data, constant=(0, 0), pad_size=14):
        H, W = data.shape[-2:]
        pad_h = (pad_size - H) % pad_size
        pad_w = (pad_size - W) % pad_size

        pad_values = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

        data = F.pad(data, pad_values, value=constant[0])

        return data, pad_values

    def unpad(self, data, pad_values):
        pad_left, pad_right, pad_top, pad_bottom = pad_values
        return data[..., pad_top:-pad_bottom, pad_left:-pad_right]

    def forward(self, x, labels_pm25=None, region_targets_pm25=None, labels_pm10=None, region_targets_pm10=None, timestamps: torch.Tensor = None, prev_vals: torch.Tensor = None):
        """
        x: (B, T, C, H, W)  # NWP outputs (T=window_size, C=n_variables)
        labels_pm25: (B, H, W)  # ground truth PM2.5 values
        region_targets_pm25: (B, R=19)  # ground truth PM2.5 values for each region
        labels_pm10: (B, H, W)  # ground truth PM10 values
        region_targets_pm10: (B, R=19)  # ground truth PM10 values for each region
        timestamps: (B, 5)  # (year, month, day, hour, lead_time)
        """
        # if self.pm25 and (labels_pm25 is None or region_targets_pm25 is None):
        #     raise ValueError("labels_pm25 and region_targets_pm25 must be provided")
        # if self.pm10 and (labels_pm10 is None or region_targets_pm10 is None):
        #     raise ValueError("labels_pm10 and region_targets_pm10 must be provided")

        
        lead_times_to_predict = torch.arange(1, self.end_lead_time + 1, device=x.device)
        B = x.shape[0]
        norm_pm_0 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()
        norm_pm_1 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()
        norm_pm_2 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()
        norm_pm_3 = torch.zeros(B, self.window_size, self.input_height * self.input_width).cuda()

        for i in range(self.window_size):
            cur_indices = np.array([4, 10, 16, 22])
            cur_sim_vals = x[:,i, cur_indices, :,:]
            norm_pm_0[:, i] = cur_sim_vals[:,0,:,:].reshape(B, self.input_height * self.input_width)
            norm_pm_1[:, i] = cur_sim_vals[:,1,:,:].reshape(B, self.input_height * self.input_width)
            norm_pm_2[:, i] = cur_sim_vals[:,2,:,:].reshape(B, self.input_height * self.input_width)
            norm_pm_3[:, i] = cur_sim_vals[:,3,:,:].reshape(B, self.input_height * self.input_width)
        
        if self.normalization_method == "Standard":
            norm_pm_0 = (norm_pm_0 - self.pm25_mean) / self.pm25_std
            norm_pm_1 = (norm_pm_1 - self.pm25_mean) / self.pm25_std
            norm_pm_2 = (norm_pm_2 - self.pm25_mean) / self.pm25_std
            norm_pm_3 = (norm_pm_3 - self.pm25_mean) / self.pm25_std
            x[:,:,24,:,:] = (x[:,:,24,:,:] - self.pm25_mean) / self.pm25_std    #### normalization for stn images
        x = x.clone()
        for i in range(self.window_size):
            x[:, i, 4, :, :] = norm_pm_0[:, i].reshape(B, self.input_height, self.input_width)
            x[:, i, 10, :, :] = norm_pm_1[:, i].reshape(B, self.input_height, self.input_width)
            x[:, i, 16, :, :] = norm_pm_2[:, i].reshape(B, self.input_height, self.input_width)
            x[:, i, 22, :, :] = norm_pm_3[:, i].reshape(B, self.input_height, self.input_width)
       
        # ipdb.set_trace()

        L = len(lead_times_to_predict)
        x = x.repeat_interleave(L, dim=0)
        x, pad_values = self.pad(x)
        
        _, T, C, H, W = x.shape
        x = x.view(B*L, -1, H, W)
        
        def condition_time(self, target_time):
            assert target_time.shape[-1] == 5
            lead_time = target_time[:, -1].to(x.device).int()
            model_time = target_time[:, 1:-1].to(x.device).int()

            lead_time_emb = self.condition_lead_time(lead_time)
            model_time_emb = torch.cat([
                emb(model_time[:, i])
                for i, emb in enumerate(self.condition_model_time)
            ])

            lead_time_emb = lead_time_emb.view(B*L, -1, 1, 1).repeat(1, 1, H, W)
            model_time_emb = model_time_emb.view(B*L, -1, 1, 1).repeat(1, 1, H, W)
            return torch.cat([lead_time_emb, model_time_emb], dim=1)

        # ipdb.set_trace()
        timestamps = timestamps[:,6,:].repeat_interleave(L, dim=0)
        cond = None
        lead_times = lead_times_to_predict.to(x.device).repeat(B) #.view(-1,1)
        cond = self.condition_lead_time(lead_times)
        timestamps = torch.cat([timestamps,lead_times.unsqueeze(-1)],dim=-1)
        # if timestamps is not None:
        #     lead_times = timestamps[:, -1].to(x.device)
        #     cond = self.condition_lead_time(lead_times) # 

        if self.concat_time_to_input:
            time_emb = condition_time(self, timestamps)
            x = torch.cat([x, time_emb], dim=1)
        
        out = self.resnet1(x, cond)
        out = self.down(out)
        out = self.vit(out, cond)
        out = self.up(out)
        out = self.resnet2(out, cond)

        out = self.unpad(out, pad_values) # (B, 128, 82, 67)

        pm_preds = self.classifier_pm25(out)
        pm_preds = pm_preds.squeeze(dim=1).reshape(B,L,self.input_height,self.input_width)
        if self.normalization_method == "Standard":
            pm_preds = pm_preds * self.pm25_std + self.pm25_mean
        return pm_preds

        # ret = OrderedDict()
        # ret["loss"] = 0
        # loss_pm25 = 0
        # loss_pm10 = 0
        # if self.pm25:
        #     nan_mask = torch.isnan(labels_pm25)
        #     labels_pm25 = torch.bucketize(labels_pm25.contiguous(), self.pm25_boundaries, right=True)
        #     labels_pm25[nan_mask] = -100
        #     logits_pm25 = self.classifier_pm25(out)
        #     loss_pm25 = F.cross_entropy(logits_pm25, labels_pm25)
        #     predicted_classes_pm25 = torch.argmax(logits_pm25, dim=1)
        #     predicted_values_pm25 = categorical_to_continuous(predicted_classes_pm25, self.pm25_boundaries)
        #     ret["logits_pm25"] = logits_pm25
        #     ret["predicted_pm25"] = predicted_values_pm25
        #     ret["loss_pm25"] = loss_pm25

        #     if self.direct_regional:
        #         if self.ignore_backbone:
        #             region_preds_pm25 = self.regr_regional_pm25(out.detach())
        #         else:
        #             region_preds_pm25 = self.regr_regional_pm25(out)
        #         region_mask = ~torch.isnan(region_targets_pm25)
        #         regr_loss_pm25 = F.mse_loss(
        #             region_targets_pm25[region_mask], 
        #             region_preds_pm25[region_mask]
        #         )
        #         ret["region_preds_pm25"] = region_preds_pm25
        #         ret["regr_loss_pm25"] = regr_loss_pm25

        # if self.pm10:
        #     nan_mask = torch.isnan(labels_pm10)
        #     labels_pm10 = torch.bucketize(labels_pm10.contiguous(), self.pm10_boundaries, right=True)
        #     labels_pm10[nan_mask] = -100
        #     logits_pm10 = self.classifier_pm10(out)
        #     loss_pm10 = F.cross_entropy(logits_pm10, labels_pm10)
        #     predicted_classes_pm10 = torch.argmax(logits_pm10, dim=1)
        #     predicted_values_pm10 = categorical_to_continuous(predicted_classes_pm10, self.pm10_boundaries)
        #     ret["logits_pm10"] = logits_pm10
        #     ret["predicted_pm10"] = predicted_values_pm10
        #     ret["loss_pm10"] = loss_pm10

        #     if self.direct_regional:
        #         if self.ignore_backbone:
        #             region_preds_pm10 = self.regr_regional_pm10(out.detach())
        #         else:
        #             region_preds_pm10 = self.regr_regional_pm10(out)
        #         region_mask = ~torch.isnan(region_targets_pm10)
        #         regr_loss_pm10 = F.mse_loss(
        #             region_targets_pm10[region_mask], 
        #             region_preds_pm10[region_mask]
        #         )
        #         ret["region_preds_pm10"] = region_preds_pm10
        #         ret["regr_loss_pm10"] = regr_loss_pm10

        # if self.direct_regional:
        #     ret["loss"] = loss_pm25 + loss_pm10 + regr_loss_pm25 + regr_loss_pm10
        # else:
        #    ret["loss"] = loss_pm25 + loss_pm10
        # return ret
    
    def get_ignore_keys_for_eval(self):
        keys = []
        if self.pm25:
            keys += ["loss_pm25", "logits_pm25"]
 
            if self.direct_regional:
                keys += ["regr_loss_pm25"]

        if self.pm10:
            keys += ["loss_pm10", "logits_pm10"]

            if self.direct_regional:
                keys += ["regr_loss_pm10"]
        return keys