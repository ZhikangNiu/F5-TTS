import math
from typing import List
from typing import Union

import numpy as np
import torch

import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import f5_tts.model.activations as activations


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        use_cuda_kernel: bool = False,
        snake_logscale: bool = True,
    ):
        super().__init__()

        self.use_cuda_kernel = use_cuda_kernel
        self.snake_logscale = snake_logscale

        self.convs1 = nn.ModuleList(
            [
                WNConv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))
                for d in dilation
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                WNConv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))
                for _ in range(len(dilation))
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.use_cuda_kernel:
            from f5_tts.model.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            from f5_tts.model.alias_free_activation.torch.act import Activation1d as TorchActivation1d

            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=activations.Snake(channels, alpha_logscale=self.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=self.snake_logscale))
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        # dilation的输出为: L_out = (L_in + 2*padding - dilation * (kernel_size -1) - 1) / stride + 1
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for block in self.convs1:
            nn.utils.remove_weight_norm(block)
        for block in self.convs2:
            nn.utils.remove_weight_norm(block)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        upsample_rates=[5, 5, 3, 2, 2, 2],
        upsample_kernel_sizes=[9, 9, 5, 4, 4, 4],
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        activation="snakebeta",
        snake_logscale="true",
        use_cuda_kernel=False,
    ):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        self.latent_dim = latent_dim
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.activation = activation
        self.snake_logscale = snake_logscale

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.use_cuda_kernel:
            from f5_tts.model.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            from f5_tts.model.alias_free_activation.torch.act import Activation1d as TorchActivation1d

            Activation1d = TorchActivation1d

        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)

        # Pre-conv
        self.conv_pre = WNConv1d(latent_dim, upsample_initial_channel, 7, 1, padding=3)

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        WNConvTranspose1d(
                            self.upsample_initial_channel // (2**i),
                            self.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    DecoderBlock(
                        ch,
                        k,
                        d,
                        activation=self.activation,
                        use_cuda_kernel=self.use_cuda_kernel,
                        snake_logscale=self.snake_logscale,
                    )
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=snake_logscale)
            if self.activation == "snake"
            else (activations.SnakeBeta(ch, alpha_logscale=snake_logscale) if self.activation == "snakebeta" else None)
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)
        self.conv_post = WNConv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)  # x: B, D, T (4,100,256) -> (4,1536,256)

        for i in range(self.num_upsamples):
            # Upsampling
            # 通道数变化：1536 → 768 → 384 → 192 → 96 → 48 → 24
            # 时间维度变化：L → 4L → 4L → 2L → 2L → 2L → 2L（假设初始长度为L）
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)  # L_out =(L_in −1)×stride+kernel_size−2×padding
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)  # B, ch, T (4,24,65536) -> B, 1, T (4,24,65536)

        x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for up_block in self.ups:
                for up_block_i in up_block:
                    nn.utils.remove_weight_norm(up_block_i)
            for block in self.resblocks:
                block.remove_weight_norm()
            nn.utils.remove_weight_norm(self.conv_pre)
            nn.utils.remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [3, 5, 5, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 5, 5, 3],
        vae_dim: Union[int, list] = 64,
        sample_rate: int = 24000,
        upsample_rates=[5, 5, 3, 2, 2, 2],
        upsample_kernel_sizes=[9, 9, 5, 4, 4, 4],
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        activation="snakebeta",
        snake_logscale="true",
        use_cuda_kernel=False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.sample_rate = sample_rate
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.vae_dim = vae_dim

        self.pre_block = nn.Linear(latent_dim, self.vae_dim)
        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_var = nn.Linear(self.vae_dim, self.vae_dim)
        self.decoder_proj = nn.Linear(self.vae_dim, latent_dim)

        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.activation = activation
        self.snake_logscale = snake_logscale
        self.use_cuda_kernel = use_cuda_kernel

        self.decoder = Decoder(
            latent_dim=self.vae_dim,
            upsample_rates=self.upsample_rates,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            upsample_initial_channel=self.upsample_initial_channel,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
            activation=self.activation,
            snake_logscale=self.snake_logscale,
            use_cuda_kernel=self.use_cuda_kernel,
        )

        # self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_kl_loss(self, mu, log_var):
        # KL Loss 公式
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var.exp() + 1e-6), dim=-1)  # 按最后一维求和
        return kl_loss.mean()  # 求 batch 的平均值

    def encode(self, audio_data: torch.Tensor):
        z = self.encoder(
            audio_data
        ).transpose(
            1, 2
        )  # torch.Size([72, 1024, 29]),[B x D x T] -> torch.Size([72, 29, 1024]),[B x T x D] ->vq torch.Size([72, 29, 8]),[B x D x T]
        z = self.pre_block(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        log_var = torch.clamp(log_var, min=-12, max=12)  # log var可能会爆掉

        z_hat = self.reparameterize(mu, log_var)
        kl_loss = self.compute_kl_loss(mu, log_var)

        return z_hat, mu, log_var, kl_loss

    def decode(self, z: torch.Tensor):
        recon = self.decoder(z.transpose(1, 2))
        return recon

    def forward(self, audio_data: torch.Tensor, sample_rate: int = None):
        audio_data = self.preprocess(audio_data, sample_rate)
        z, mu, log_var, kl_loss = self.encode(audio_data)

        audio = self.decode(z)
        return audio
        # return {
        #     "audio": x[..., :length],
        #     "z": z,
        #     "mu": mu,
        #     "log_var": log_var,
        #     "vae/kl_loss": kl_loss,
        # }

    @torch.no_grad()
    def extract_latent(self, audio_data):
        audio_data = self.preprocess(audio_data, self.sample_rate)
        z, mu, log_var, kl_loss = self.encode(audio_data)
        return z

    @staticmethod
    def load_encoder_from_pretrain(model_path, **kwargs):
        model_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model = AutoencoderKL(**kwargs).cuda()
        model.load_state_dict(model_dict)
        del model.decoder
        model.eval()
        return model

    @staticmethod
    def load_decoder_from_pretrain(model_path, **kwargs):
        model_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model = AutoencoderKL(**kwargs).cuda()
        model.load_state_dict(model_dict)
        del model.encoder
        model.eval()
        return model

    @staticmethod
    def load_from_pretrain(model_path, **kwargs):
        model_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model = AutoencoderKL(**kwargs).cuda()
        model.load_state_dict(model_dict)
        model.eval()
        return model
