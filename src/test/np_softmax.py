"""
softmax函数是将模型直接的logit输出转化为和为1的概率分布
x_i为模型输出 例如为多分类模型的输出 那么再套一层softmax即为
softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
"""

import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    # 输入为二维ndarray数组 （batch，num_classes;(3,4) 3个样本，每个样本4个类别，输出4个类别的logits）

    max_x = np.max(x, axis=1, keepdims=True)

    exp_x = np.exp(x - max_x) # (3,4), 期望x - max_x能自动广播机制，max_x自动补全为（3,4），每一行都是该行的最大值


    row_sum = np.sum(exp_x, axis=1, keepdims=True) # (3,1)

    return exp_x / row_sum


input_array = np.array([[1,2,5,9],[1,5,8,9],[4,8,9,6]], dtype=np.float32)

def main():
    print(softmax(input_array))
    return

if __name__ == "__main__":
    main()

