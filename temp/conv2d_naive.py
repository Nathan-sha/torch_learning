import numpy as np


np.random.seed(42)

class conv2d:
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0):
        self.in_channel, self.out_channel = in_channel, out_channel
        self.kernel_size, self.stride, self.padding = kernel, stride, padding

        # 初始化W.shape [c_out, c_in, k, k]
        self.W = np.random.randn(out_channel, in_channel, self.kernel_size, self.kernel_size)
        self.b = np.zeros(out_channel)


    def forward(self, x: np.ndarray):
        N, c_in, h, w = x.shape
        # 计算out size
        h_out = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        w_out = h_out

        # padding
        x_pad = np.pad(x, pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                       mode="constant", constant_values=0)
        
        out = np.zeros((N, self.out_channel, h_out, w_out), dtype=np.float32)

        for n in range(N):
            for c_out in range(self.out_channel):
                for i in range(h_out):
                    for j in range(w_out):
                        i_start = i*self.stride
                        j_start = j*self.stride
                        window = x_pad[n, :, i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size]
                        out[n, c_out, i, j] = np.sum(self.W[c_out] * window) + self.b[c_out]   

        return out # [N, c_out, h_out, w_out]
    

import torch
import torch.nn as nn

def main():
    x = np.random.rand(5, 3, 6, 6) # N, C, H, W
    c_out = 9
    N, c_in, h, w = x.shape
    kernel_size = 3
    padding = 1
    stride = 2


    conv2d_np = conv2d(c_in, c_out, kernel_size, stride, padding)
    conv2d_torch = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
    # 同步numpy权重给torch
    conv2d_torch.weight.data = torch.from_numpy(conv2d_np.W)
    conv2d_torch.bias.data = torch.from_numpy(conv2d_np.b)

    out_np = conv2d_np.forward(x)
    out_torch = conv2d_torch(torch.from_numpy(x))

    print("out_np size", out_np.shape)
    print("out_torch size", out_torch.shape)
    print("max error", torch.max(torch.abs(torch.from_numpy(out_np) - out_torch)))

    return

if __name__ == "__main__":
    main()