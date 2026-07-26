import numpy as np

class BN_numpy:
    def __init__(self, num_feature=512, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((1, num_feature), dtype=np.float32)
        self.beta = np.zeros((1, num_feature), dtype=np.float32)

        self.running_mean = np.zeros((1, num_feature))
        self.running_var = np.ones((1, num_feature))

        self.x = None
        self.mean = None
        self.var = None
        self.std = None
        self.x_hat = None
        self.dgamma = None
        self.dbeta = None
        self.training = True

    def forward(self, x):
        self.x = x
        if self.training:
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.std = np.sqrt(self.var + self.eps)
            self.x_hat = (x - self.mean) / self.std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum)*self.mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum)*self.var
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dx = dout * self.gamma / self.std
        return dx

    def zero_grad(self):
        self.dgamma = None
        self.dbeta = None

    def train(self): self.training = True
    def eval(self): self.training = False

class LN_numpy:
    def __init__(self, num_feature=512, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, num_feature), dtype=np.float32)
        self.beta = np.zeros((1, num_feature), dtype=np.float32)
        self.x = None
        self.mean = None
        self.var = None
        self.std = None
        self.x_hat = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        axis = tuple(range(len(dout.shape)-1))
        self.dbeta = np.sum(dout, axis=axis, keepdims=True)
        self.dgamma = np.sum(dout * self.x_hat, axis=axis, keepdims=True)
        dx = dout * self.gamma / self.std
        return dx

    def zero_grad(self):
        self.dgamma = None
        self.dbeta = None

class RMSNorm_numpy:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.x = None
        self.rms = None
        self.dgamma = None

    def _rms(self, x):
        return np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def forward(self, x):
        self.x = x
        self.rms = self._rms(x)
        return x / self.rms * self.gamma

    def backward(self, dout):
        axis = tuple(range(len(dout.shape)-1))
        self.dgamma = np.sum(dout * self.x / self.rms, axis=axis, keepdims=True)
        dx = (self.gamma/self.rms) * (dout - (self.x/self.rms)*np.sum(dout*self.x/self.rms, axis=-1, keepdims=True)/self.gamma.shape[-1])
        return dx

    def zero_grad(self):
        self.dgamma = None