from tensorboardX import SummaryWriter
import os

from utils.timestamp import get_strftime
from utils.make_dirs import make_dir

def get_summary_writer(root_path):
    path=os.path.join(root_path,get_strftime())
    make_dir(path)
    writer = SummaryWriter(path)
    return writer
 