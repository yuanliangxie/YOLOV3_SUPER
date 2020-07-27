import os
import traceback

class log_recorder(object):
    """Save training process to log file with simple plot function."""
    def __init__(self, file_path, resume=False):
        self.file = None
        self.resume = resume
        if os.path.isfile(file_path):
            if resume:
                self.file = open(file_path, "a")
            else:
                self.file = open(file_path, 'w')
        else:
            self.file = open(file_path, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
                self.file.write(target_str+'\n')
                self.file.flush()
        else:
            print(target_str)
            self.file.write(target_str+'\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

class print_logger(object):#它的功能只是用在测试上
    def __init__(self):
        pass
    def append(self,target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
        else:
            print(target_str)