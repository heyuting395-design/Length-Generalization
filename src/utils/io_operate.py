import os
import yaml
import json
import re
import torch
import logging
import numpy as np
from datetime import datetime
from _ctypes import PyObj_FromPtr

# -----------------------------
# 1. JSON 特殊格式化 (保留你的精华)
# -----------------------------
class NoIndent(object):
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))
    def __init__(self, **kwargs):
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC
        json_repr = super(MyEncoder, self).encode(obj)
        for match in self.regex.finditer(json_repr):
            id_val = int(match.group(1))
            no_indent = PyObj_FromPtr(id_val)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)
            json_repr = json_repr.replace('"{}"'.format(format_spec.format(id_val)), json_obj_repr)
        return json_repr

# -----------------------------
# 2. 核心 IO 控制器
# -----------------------------
class IOManager:
    """
    严谨的实验 IO 管理器：
    - 自动创建时间戳目录
    - 自动备份配置文件
    - 统一管理 Log 和 Checkpoint
    - 支持 YAML 和 JSON 格式配置
    """
    def __init__(self, config_path, exp_root="outputs"):
        # 1. 智能加载原始配置 (支持 yaml 和 json)
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                self.config_dict = json.load(f)
            else:
                self.config_dict = yaml.load(f, Loader=yaml.FullLoader)
                
        # 增加一个别名，兼容 train.py 里的调用 (io.config)
        self.config = self.config_dict 
        
        # 2. 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config_dict.get('exp_name', 'default_exp')
        self.exp_dir = os.path.join(exp_root, f"{exp_name}_{timestamp}")
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.fig_dir = os.path.join(self.exp_dir, "figures")
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        # 3. 备份配置 (根据原后缀名备份)
        backup_path = os.path.join(self.exp_dir, f"config_backup{os.path.splitext(config_path)[1]}")
        if backup_path.endswith('.json'):
            self.save_json_noindent(backup_path, self.config_dict)
        else:
            self.save_yaml(backup_path, self.config_dict)
        
        # 4. 初始化日志
        self.logger = self._setup_logger()

    def _setup_logger(self):
        log_path = os.path.join(self.exp_dir, "train.log")
        logger = logging.getLogger(self.exp_dir) # 唯一标识
        logger.setLevel(logging.DEBUG)
        
        fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # 屏幕输出
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        # 文件输出
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(fmt)
        
        # 避免重复打印
        if not logger.handlers:
            logger.addHandler(sh)
            logger.addHandler(fh)
            
        return logger

    def save_yaml(self, path, data):
        with open(path, 'w', encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    def save_json_noindent(self, path, data):
        """保存为易读的 JSON，内部列表不换行"""
        # 预处理：将 ndarray 或 list 包装进 NoIndent
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (list, np.ndarray)):
                processed_data[k] = NoIndent(v.tolist() if isinstance(v, np.ndarray) else v)
            else:
                processed_data[k] = v
                
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(processed_data, cls=MyEncoder, indent=2))

    def read_json_data(self, path):
        """读取标准的 JSON 文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON data file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_model(self, model, epoch, is_best=False):
        name = "best_model.pth" if is_best else f"ckpt_epoch_{epoch}.pth"
        path = os.path.join(self.ckpt_dir, name)
        torch.save(model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def get_path(self, sub_dir, filename):
        """获取实验目录下子目录的路径"""
        target_dir = os.path.join(self.exp_dir, sub_dir)
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, filename)
def read_json_data(path):
    """一个独立的工具函数，方便其他模块直接调用"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON data file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)