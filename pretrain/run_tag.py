""" tag = (lr, B, T, max_steps) 的函数。 train.py 落盘、sample.py
加载共用这一处 """

def make_tag(config):
    return f"lr{config['lr']}_B{config['B']}_T{config['T']}_S{config['max_steps']}"
