import numpy as np


def matrix_multiplier(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # m*k @ k*n

    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"无法相乘: A 的列数 {A.shape[1]} != B 的行数 {B.shape[0]}"
        )
    m,k,n = A.shape[0], A.shape[1], B.shape[1]

    if B.shape[0] != k:
        return np.zeros((m,n), dtype=A.dtype)

    res = np.zeros((m,n), dtype=A.dtype)
    
    for i in range(m):
        for j in range(n):
            for p in range(k):
                res[i][j] += A[i][p] * B[p][j]

    return res
    

A = np.array([[1,2,3],[4,5,9],[6,9,7],[1,2,9]], dtype=np.float32) # 4*3
B = np.array([[1,3],[6,9],[2,9],[1,2]], dtype=np.float32) # 3*2

def main():
    print(A,"@",B)
    print(matrix_multiplier(A,B),"matrix_multiplier")
    print(A@B,"np.matmul")
    return

if __name__ == "__main__":
    main()