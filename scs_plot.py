"""scs_plot.py — 四张复现图的统一绘图风格与保存工具。"""
import os
import matplotlib
matplotlib.use('Agg')          # 纯 CPU、无显示环境
import matplotlib.pyplot as plt

OUTDIR = 'outputs'


def setup():
    """统一 matplotlib 风格(贴近论文/原仓库:浅网格、加粗轴标签、合适字号)。"""
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'axes.axisbelow': True,
        'legend.framealpha': 0.9,
    })


def save(fig, name):
    """保存到 outputs/<name>.png 并关闭。"""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name if name.endswith('.png') else name + '.png')
    fig.savefig(path)
    plt.close(fig)
    return path
