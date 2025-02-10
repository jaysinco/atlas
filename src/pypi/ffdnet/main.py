import numpy as np
import torch.nn as nn
import basicblock as B
import torch
import os
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------
# FFDNet
# --------------------------------------------
class FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = B.PixelUnShuffle(upscale_factor=sf)

        m_head = B.conv(in_nc*sf*sf+1, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc*sf*sf, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma: torch.Tensor):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x


def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def imshow(bef, aft):
    fig = plt.figure()
    ax = fig.subplots(1, 2)
    ax[0].imshow(np.squeeze(bef), interpolation='nearest')
    ax[1].imshow(np.squeeze(aft), interpolation='nearest')
    plt.show()


def tensor2uint(img: torch.Tensor):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


if __name__ == '__main__':
    noise_level = 8
    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_file_dir, "data")

    model = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R')
    print(describe_model(model))

    model_path = os.path.join(data_dir, "ffdnet_color_clip.pth")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    img_path = os.path.join(data_dir, "4.jpg")
    img_bef = imread_uint(img_path, n_channels=3)
    img_L = np.float32(img_bef/255.)
    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0)
    sigma = torch.full((1,1,1,1), noise_level/255.).type_as(img_L)
    img_E = model(img_L, sigma)
    img_aft = tensor2uint(img_E)

    imshow(img_bef, img_aft)
    # cv2.imwrite(os.path.join(data_dir, "out.jpg"), img_aft[:, :, [2, 1, 0]])


    # x = torch.randn((1,3,1232,1080))
    # noise_level_img = 15
    # sigma = torch.full((1,1,1,1), noise_level_img/255.)
    # x = model(x, sigma)
    # print(x.shape)
