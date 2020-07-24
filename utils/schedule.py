import numpy as np


class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr

class StepDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, gamma, decay_step):
        super().__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        if decay_step == None:
            self.__decay_step =[int(T_max*0.8), int(T_max * 0.9)]
        else:
            self.__decay_step =[decay_step[0], decay_step[1]]
        self.gamma = gamma
        self.lr = lr_init

    def step(self, t):
        if  t < self.__decay_step[0]:
            lr = self.lr

        elif t >= self.__decay_step[0] and t <= self.__decay_step[1] :
            lr = self.lr*self.gamma

        else:
            lr = self.lr*(self.gamma**2)
        for param_group in self.__optimizer.param_groups:
            param_group['lr'] = lr