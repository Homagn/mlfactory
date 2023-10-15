# image + time step conditioned xunet for diffusion models
# attempt to reproduce something like https://arxiv.org/pdf/2210.04628.pdf (figure 4)
# using a lot of code from - https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py


import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)



class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, x_cond):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)

        x_cond = x_cond.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_lnc = self.ln(x_cond)

        # example of mha(query , key , value)
        '''
        For example, in machine translation, the query could be a word in the source sentence, 
        the key could be a word in the target sentence, and the value could be the translation of the word in the source sentence.
        '''
        attention_value, _ = self.mha(x_ln, x_lnc, x_lnc) #in cross attention we just need to change the key and value
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.ca1 = CrossAttention(128)

        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.ca2 = CrossAttention(256)

        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)
        self.ca3 = CrossAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.ca4 = CrossAttention(128)

        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.ca5 = CrossAttention(64)

        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.ca6 = CrossAttention(64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t) #shape - ([None, 128, 32, 32])
        
        x2 = self.sa1(x2) ##shape - ([None, 128, 32, 32])
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t) #
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

    def unet_cond_forward(self, x, xc, t):
        # x is the sampled noise being purified
        # xc is the image conditioning being applied

        x1 = self.inc(x)
        x1c = self.inc(xc)

        x2 = self.down1(x1, t)
        x2c = self.down1(x1c, t)

        #print("shapes before attention ",x2.shape, x2c.shape)

        x2 = self.sa1(x2)
        x2c = self.sa1(x2c)

        #print("shapes after attention ",x2.shape, x2c.shape)

        #apply cross attention now to exchange information
        x_xc1 = self.ca1(x2, x2c)
        xc_x1 = self.ca1(x2c, x2)

        x3 = self.down2(x_xc1, t)
        x3c = self.down2(xc_x1, t)

        x3 = self.sa2(x3)
        x3c = self.sa2(x3c)

        #apply cross attention now to exchange information
        x_xc2 = self.ca2(x3, x3c)
        xc_x2 = self.ca2(x3c, x3)

        x4 = self.down3(x_xc2, t)
        x4c = self.down3(xc_x2, t)

        #apply cross attention now to exchange information
        x_xc3 = self.ca3(x4, x4c)
        xc_x3 = self.ca3(x4c, x4)


        #reached bottom if Unet
        x4 = self.bot1(x_xc3)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x4c = self.bot1(xc_x3)
        if not self.remove_deep_conv:
            x4c = self.bot2(x4c)
        x4c = self.bot3(x4c)


        x = self.up1(x4, x3, t)
        xc = self.up1(x4c, x3c, t)

        x = self.sa4(x)
        xc = self.sa4(xc)

        #apply cross attention now to exchange information
        x_xc4 = self.ca4(x, xc)
        xc_x4 = self.ca4(xc, x)



        x = self.up2(x, x2, t)
        xc = self.up2(xc, x2c, t)

        x = self.sa5(x)
        xc = self.sa5(xc)

        #apply cross attention now to exchange information
        x_xc5 = self.ca5(x, xc)
        xc_x5 = self.ca5(xc, x)



        x = self.up3(x_xc5, x1, t)
        xc = self.up3(xc_x5, x1, t)

        x = self.sa6(x)
        xc = self.sa6(xc)

        #apply cross attention now to exchange information
        x_xc6 = self.ca6(x, xc)
        xc_x6 = self.ca6(xc, x)


        output = self.outc(x_xc6) #dont care about xc_x6
        return output


    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional_label(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)

class UNet_conditional_labelimage(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, xc, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        #return self.unet_forwad(x, t)
        return self.unet_cond_forward(x, xc, t)


if __name__ == '__main__':
    channels_in = 3
    channels_out = 1 # can use 1 for example when predicting depth maps
    img_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 2 #batch_size
    i = 20 # anything between 0 and number of noise steps (generally 1000)
    num_classes = 36 # can denote 36 different camera poses in a fixed order
    # example say we have classes from 1 to 10 and we are passing 2 labels denoted by class 6 and class 6
    labels = torch.Tensor([6] * n).long().to(device)
    x = torch.randn((n, channels_in, img_size, img_size)).to(device)
    t = (torch.ones(n) * i).long().to(device)

    #dummy conditional image
    xc = torch.randn((n, channels_in, img_size, img_size)).to(device)


    '''
    print("Testing Unet conditioned on label only ")
    model = UNet_conditional_label(c_in = channels_in, c_out = channels_out, num_classes= num_classes).to(device)
    print("Num params in millions: ", sum(p.numel() for p in model.parameters())/1e6)

    predicted_noise = model(x, t, labels)
    print("got predicted noise ",predicted_noise.shape)
    '''



    print("Testing Unet conditioned on both label and reference image ")
    model = UNet_conditional_labelimage(c_in = channels_in, c_out = channels_out, num_classes= num_classes).to(device)
    print("Num params in millions: ", sum(p.numel() for p in model.parameters())/1e6)

    predicted_noise = model(x, xc, t, labels)
    print("got predicted noise ",predicted_noise.shape)


'''
#diffusion model training idea

1. sample target image (to be generated from noise) and reference image which is not drastically different angle (slight angle variations)

2. pass the embedding of the angle difference as label and the reference image and generate target

3. stochastic conditioning idea- sometimes you can chose reference image at widely different angle from target image but then randomly 
insert intermediate pose images as additional reference images in between
'''