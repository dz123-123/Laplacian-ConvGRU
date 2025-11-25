from blessings import Terminal
import progressbar
import sys


class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        #总训练周期数
        self.n_epochs = n_epochs
        #训练数据集的大小
        self.train_size = train_size
        #验证数据集大小
        self.valid_size = valid_size
        #创建一个终端操作
        self.t = Terminal()
        s = 10
        e = 1   # epoch bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        h = self.t.height

        for i in range(10):
            print('')
        self.epoch_bar = progressbar.ProgressBar(maxval=n_epochs, fd=Writer(self.t, (0, h-s+e)))

        self.train_writer = Writer(self.t, (0, h-s+tr))
        self.train_bar_writer = Writer(self.t, (0, h-s+tr+1))

        self.valid_writer = Writer(self.t, (0, h-s+ts))
        self.valid_bar_writer = Writer(self.t, (0, h-s+ts+1))

        self.reset_train_bar()
        self.reset_valid_bar()

    def reset_train_bar(self):
        self.train_bar = progressbar.ProgressBar(maxval=self.train_size, fd=self.train_bar_writer).start()

    def reset_valid_bar(self):
        self.valid_bar = progressbar.ProgressBar(maxval=self.valid_size, fd=self.valid_bar_writer).start()


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
'''
class AverageMeter(object):  
    """Computes and stores the average and current value"""  

    def __init__(self, i=1, precision=3):  
        # 初始化方法  
        # i: 计量器的数量，默认为1  
        # precision: 输出时浮点数的精度，默认为3  
        self.meters = i  # 存储计量器的数量  
        self.precision = precision  # 存储输出时的精度  
        self.reset(self.meters)  # 调用reset方法初始化计量器相关的变量  

    def reset(self, i):  
        # 重置方法，用于初始化或重新初始化计量器  
        # i: 计量器的数量  
        self.val = [0]*i  # 当前值列表，每个计量器一个值，初始化为0  
        self.avg = [0]*i  # 平均值列表，初始化为0  
        self.sum = [0]*i  # 总和列表，用于计算平均值，初始化为0  
        self.count = 0  # 计数，记录更新了多少次（不考虑批量大小）  

    def update(self, val, n=1):  
        # 更新方法，用于根据新的值更新计量器  
        # val: 可以是单个值或值的列表，如果是单个值，则转换为列表  
        # n: 更新时考虑的批量大小（默认为1），影响总和的计算  
        if not isinstance(val, list):  
            val = [val]  # 如果不是列表，则转换为列表  
        assert(len(val) == self.meters)  # 确保提供的值的数量与计量器数量一致  
        self.count += n  # 增加计数  
        for i, v in enumerate(val):  
            self.val[i] = v  # 更新当前值（注意：这里可能不是典型的“当前值”用法，通常只记录最近一次的值）  
            self.sum[i] += v * n  # 更新总和，考虑批量大小  
            self.avg[i] = self.sum[i] / self.count  # 更新平均值  

    def __repr__(self):  
        # 字符串表示方法，用于打印类的实例信息  
        # 格式化当前值和平均值，根据初始化时设置的精度  
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])  # 格式化当前值列表  
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])  # 格式化平均值列表  
        return '{} ({})'.format(val, avg)  # 返回包含当前值和平均值的字符串
'''