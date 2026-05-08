import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def apply_plot_settings():
    """
    应用专业科研绘图风格，适配 Nature/Science 风格建议
    """
    plt.rcParams.update({
        # 字体：优先使用 Times New Roman (Windows/Linux) 或 serif
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 11,
        
        # 坐标轴
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True, # 网格线置于底层
        
        # 刻度
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        
        # 线条与标记
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.markeredgewidth": 1.0,
        
        # 图例
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False, # 使用直角边框更显硬核
        
        # 网格与背景
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        
        # 输出
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf" # 建议保存为矢量图
    })

def get_color_palette():
    """
    返回符合学术审美的配色方案
    包含：宝石蓝、洋红、翡翠绿、深金、深紫色等
    """
    return {
        "royal_blue": "#002060",  # 宝石深蓝 (极高对比度)
        "crimson": "#C00000",     # 洋红/深红 (关键曲线)
        "emerald": "#008B45",     # 翡翠绿 (IID 或 Baseline)
        "gold": "#B8860B",        # 暗金 (特殊标注)
        "purple": "#68228B",      # 深紫色
        "teal": "#008080",        # 深青色
        "slate": "#2F4F4F"        # 石板灰
    }

def get_line_styles():
    """
    返回线型循环，用于多曲线对比
    """
    return ["-", "--", "-.", ":"]

def get_functional_colors():
    """
    为特定任务分配的功能性配色（基于宝石蓝和洋红）
    """
    colors = get_color_palette()
    return {
        "train": colors["slate"],      # 训练损失用中性色
        "iid": colors["emerald"],      # IID 用稳重的绿色
        "ood": colors["crimson"],      # OOD 用醒目的洋红
        "primary": colors["royal_blue"],# 核心模型用宝石蓝
        "secondary": colors["gold"]    # 对比模型用暗金色
    }