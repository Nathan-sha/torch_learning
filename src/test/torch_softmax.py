
import torch
import numpy as np


from np_softmax import softmax

input_np_array = np.array([[1,2,5,9],[1,5,8,9],[4,8,9,6]], dtype=np.float32)

input_tensor = torch.tensor([[1,2,5,9],[1,5,8,9],[4,8,9,6]], dtype=torch.float32)

def main():
    print(torch.softmax(input_tensor, dim=1), "torch version")
    print(softmax(input_np_array), "np version")
    return

if __name__ == "__main__":
    main()