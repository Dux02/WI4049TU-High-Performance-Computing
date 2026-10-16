from pathlib import Path
import re
import plotext as terminal_plot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path("plots")
output.mkdir(exist_ok=True)

text = Path("node_speed.out").read_text()

def extract(label, section):
    match = re.search(rf"{re.escape(label)}:\s*([\d.eE+-]+)", section)
    if not match:
        raise ValueError(f"Cannot find {label} in node_speed.out")
    return float(match.group(1))

# Sections before the core-count sweep contain the single-core results.
before_scaling = text.split("Cores =")[0]
peak = extract("MFlops/s", before_scaling) / 1000
bandwidth = extract("MByte/s", before_scaling) / 1000

cores = []
bandwidths = []

for match in re.finditer(
    r"Cores\s*=\s*(\d+)(.*?)(?=Cores\s*=|\Z)",
    text,
    flags=re.S,
):
    cores.append(int(match.group(1)))
    bandwidths.append(extract("MByte/s", match.group(2)) / 1000)

if not cores:
    raise ValueError("No core-count measurements found")

# Bandwidth scaling.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(cores, bandwidths, "o-", label="Measured DAXPY bandwidth")
ax.set(
    xlabel="Number of cores",
    ylabel="Memory bandwidth [GB/s]",
    title="DAXPY memory bandwidth — compute-p2",
)
ax.set_xticks(cores)
ax.tick_params(axis="x", labelsize=8)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(output / "bandwidth.png", dpi=200)
fig.savefig(output / "bandwidth.pdf")
plt.close(fig)

# Assignment's ideal cache model for the current SOR update:
# 16 FLOPs; 3 double loads + 1 double store.
intensity = 16 / 32
balance = peak / bandwidth
ceiling = min(peak, bandwidth * intensity)

x = np.logspace(-2, 2, 500)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(
    x, np.minimum(peak, bandwidth * x),
    linewidth=2, label="DAXPY-based roofline",
)
ax.axvline(
    intensity, color="tab:orange", linestyle="--",
    label=f"SOR model: {intensity:.2f} FLOP/byte",
)
ax.plot(
    intensity, ceiling, "o", color="tab:orange",
    label=f"SOR model ceiling: {ceiling:.2f} GFLOP/s",
)
ax.set(
    xlabel="Arithmetic intensity [FLOP/byte]",
    ylabel="Performance [GFLOP/s]",
    title="Single-core roofline — compute-p2",
)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(output / "roofline.png", dpi=200)
fig.savefig(output / "roofline.pdf")
plt.close(fig)

print(f"Single-core DAXPY peak: {peak:.3f} GFLOP/s")
print(f"Single-core bandwidth: {bandwidth:.3f} GB/s")
print(f"Machine balance:       {balance:.3f} FLOP/byte")
print(f"SOR model intensity:   {intensity:.3f} FLOP/byte")
print(f"SOR model ceiling:     {ceiling:.3f} GFLOP/s")
print("Saved bandwidth and roofline plots in plots/")
fig = terminal_plot.figure
fig.clear()

signal = fig.signal(cores, bandwidths).lines()
fig.draw(signal)

fig.title("DAXPY memory bandwidth")
fig.label("Number of cores", axis="x")
fig.label("Bandwidth [GB/s]", axis="y")
fig.show()
