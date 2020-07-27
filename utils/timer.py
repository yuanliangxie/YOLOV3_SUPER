import time
import datetime


class Timer(object):
    '''
    A simple timer.
    '''

    def __init__(self, len_dataloder, epoch):
        self.spended_time = 0
        self.haved_steped = 0
        self.sum_step = epoch * len_dataloder

    def add(self, dur):
        self.spended_time += dur
        self.haved_steped +=10

    def get_remain_time(self, current_step):
        each_step_time = self.spended_time / self.haved_steped
        left_step = self.sum_step - current_step
        left_time = left_step * each_step_time
        return str(datetime.timedelta(seconds=int(left_time)))