import numpy as np

def conv2d(input: np.ndarray, weight: np.ndarray, stride=1, padding=0):
    batch, h_in, w_in, c_in = input.shape
    k, _, _, c_out = weight.shape

    h_out = (h_in + 2 * padding -k) // stride + 1
    w_out = (w_in + 2 * padding - k) // stride + 1

    # 3. 填充（给输入上下左右补0）
    x_pad = np.pad(input, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
    
    # 4. 初始化输出
    outputs = np.zeros((batch, h_out, w_out, c_out), dtype=np.float32)

    for b in range(batch):
        for h in range(h_out):
            for w in range(w_out):
                # 定位输入上的感受野区域
                h_start = h * stride
                h_end = h_start + k
                w_start = w * stride
                w_end = w_start + k
                
                # 取出局部感受野 [K, K, C_in]
                local_region = x_pad[b, h_start:h_end, w_start:w_end, :]


                for c in range(c_out):
                    outputs[b, h, w, c] = np.sum(local_region * weight[:,:,:,c])

    return outputs 




