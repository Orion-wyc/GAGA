import time
import os
import datetime
import json
from torch.utils.tensorboard import SummaryWriter

_b_colors = {'header': '\033[95m',
             'red': '\033[31m',
             'green': '\033[32m',
             'yellow': '\033[33m',
             'blue': '\033[34m',
             'magenta': '\033[35m',
             'cyan': '\033[36m',
             'bold': '\033[1m',
             'underline': '\033[4m'}


def print_msg(msg, style=''):
    """
        This function provides formatted print() in terminal.
    """
    current_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
    if not style or style == 'black':
        print('[{}] {}'.format(current_time, msg))
    else:
        print("[{curr_t}] {color1}{msg}{color2}".format(curr_t=current_time, color1=_b_colors[style],
                                                        msg=msg, color2='\033[0m'))


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name
        self.tic = 0.0
        self.toc = 0.0

        # 累计计时次数
        self.cnt = 0
        # 总计
        self.total_time = 0.0
        # start与end配对出现
        self.pair_flag = True

    def start(self):
        assert self.pair_flag == True, 'The amount of timer.start() and timer.end() should be the same.'
        self.tic = time.time()
        self.pair_flag = False

    def end(self):
        assert self.pair_flag == False, 'Using timer.start before timer.end'
        self.toc = time.time()
        self.total_time += self.toc - self.tic
        self.cnt += 1
        self.pair_flag = True

    @property
    def avg_time(self):
        return self.total_time / self.cnt


class SummaryBox:
    def __init__(self, task_name, log_dir=None, flush_secs=60):
        self.task_name = task_name

        # 这个时间会用作为全局log的时间
        self.start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join('runs', f"{self.start_wall_time}_{self.task_name}")
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.log_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)

        print(f"[{self.__class__.__name__}] Storing results to {self.log_dir}")

    def update_metrics(self, results, global_step=None):
        fields = results._fields
        for field, value in zip(fields, results):
            tag = f"metrics/{field}"
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)

    def update_loss(self, value, mode='train', global_step=None):
        tag = f"loss/{mode}_loss"
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)

    def add_figure(self, figure, fig_name, global_step=None):
        tag = f"figures/{fig_name}"
        self.writer.add_figure(tag=tag, figure=figure, global_step=global_step)

    def add_graph(self, model, input_to_model):
        tag = f"graphs/{model.__class__.__name__}"
        self.writer.add_graph(model, input_to_model)

    def save_config(self, configs):
        file_path = os.path.join(self.log_dir, 'config.json')
        with open(file_path, 'w+') as f:
            json.dump(configs, f, sort_keys=True, indent=4)

    def close(self):
        self.writer.close()
