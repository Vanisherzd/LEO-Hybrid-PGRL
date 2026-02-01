import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    """
    Sets a professional, academic style for all plots.
    Compatible with IEEE/Academic publication standards.
    """
    sns.set_theme(style="whitegrid")
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })
    
    print("Academic plotting style (IEEE Compliance) enabled.")

def get_color_palette():
    """Returns a high-contrast academic palette."""
    return sns.color_palette("deep")
