from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import shutil
import torch

from torch.utils.tensorboard import SummaryWriter

USE_TENSORBOARD = True


def mkdir(path, rm=False):
    if os.path.exists(path):
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/logs_{}'.format(time_str)
        mkdir(log_dir)
        self.log_path = log_dir
        if USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=log_dir)

        log_file = log_dir + '/log.txt'
        print("log file will be saved to {}".format(log_file))
        self.log = open(log_file, 'w')
        shutil.copyfile(file_name, log_dir + "/opt.txt")
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()
        if USE_TENSORBOARD:
            self.writer.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)


if __name__ == "__main__":
    from torchvision.models import resnet18

    model = resnet18(pretrained=False)


    class Opt:
        def __init__(self):
            self.save_dir = "./"


    logger = Logger(opt=Opt())
    logger.writer.add_graph(model, input_to_model=torch.rand([1, 3, 224, 224]))
    logger.close()
