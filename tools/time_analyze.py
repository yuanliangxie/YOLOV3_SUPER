from functools import wraps
import cProfile
from line_profiler import LineProfiler
import time


def print_run_time(func):
    """
    一个装饰器
    function：打印函数的运行时间
    """
    def wrapper(*args, **kw):
        local_time = time.time()
        results = func(*args, **kw)
        run_time = time.time() - local_time
        print('current Function [%s] run time is %.2f' % (func.__name__ , run_time) )
        return results
    return wrapper


def func_cprofile(f):
    """
    内建分析器
    """
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = f(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')

    return wrapper


def func_line_time(func):
    """
    这个装饰器最为有用
    function：统计被装饰的函数中每条语句的运行时间！
    """
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        lp_wrapper = profiler(func)
        results = lp_wrapper(*args, **kwargs)
        profiler.print_stats()
        return results
    return wrapper
