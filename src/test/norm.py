
import numpy as np
import torch




def bn(x: np.ndarray, gamma, beta) -> np.ndarray:
    # batchnorm 按通道C归一化，batch，以及除了特征维度以外的维度
    x_mean = x.mean(axis=0, keepdims=True)

    # x_var = x.var(axis=0, keepdims=True)
    x_var = np.sum((x - x_mean)**2, axis=0, keepdims=True) / x.shape[0]

    epsilon = 1e-5

    x_hat = (x - x_mean) / np.sqrt(x_var + epsilon)

    out = gamma * x_hat + beta

    return out

def BatchNormForward(x, ):

    return


def ln(x: np.ndarray) -> np.ndarray:
    # LayerNorm 按样本归一化

    

    return


input_array = np.array([[1,2,5,9],[1,5,8,9],[4,8,9,6]], dtype=np.float32)
input_tensor = torch.tensor(input_array)

def main():
    res_np = bn(input_array)
    print(res_np)
    # res = torch.batch_norm(input_tensor)
    # print(res)
    return

if __name__ == "__main__":
    main()