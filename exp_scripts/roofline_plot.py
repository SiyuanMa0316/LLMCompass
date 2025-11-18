import numpy as np, matplotlib.pyplot as plt

peak, bw = 370, 2 # TFLOP/s, TB/s
oi = np.logspace(-2, 4, 400)
roof = np.minimum(bw * oi, peak)
oi_knee = peak / bw

fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.loglog(oi, bw * oi, '--', lw=1, label=f'{bw} TB/s × OI')
ax.loglog(oi, [peak]*len(oi), '-', lw=1.2, label=f'{peak} TFLOP/s')
ax.loglog(oi, roof, '-', lw=1.8, label='Roofline')

# Knee and example points

ax.axvline(oi_knee, ls=':', lw=0.8)
ax.text(oi_knee*1.1, peak*10, f'OI={oi_knee:.0f}', rotation=90, va='center', ha='left', fontsize=7)

# x = [0.1,1,10,100,1e3]
# y = [min(bw*xi, peak) for xi in x ]

x= [
    0.9998372661,
    0.9917675545,
    0.9917675545,
    0.9998982851,
    0.9998982851,
    0.9995119571,
    0.9912875121,
    0.9996949173,
    0.999755919,
    0.9989025729,
    0.9998430771,
    0.9987807852
]
y=[
    1.363385499,
    0.04443118644,
    0.04405781513,
    1.390057022,
    1.410838066,
    1.271001212,
    0.01175533632,
    1.245062412,
    1.249699516,
    0.8603700513,
    1.318815407,
    0.5377312821,
]

x_prefill=[
    877.7142857,
    107.7894737,
    107.7894737,
    927.3962264,
    927.3962264,
    682.6666667,
    102.4,
    780.1904762,
    819.2,
    481.8823529,
    882.2153846,
    455.1111111,
]
y_prefill=[
    211.5023906,
    40.06499343,
    33.34601938,
    222.3971271,
    222.4491208,
    136.727968,
    17.20740103,
    206.7995087,
    204.1576849,
    159.5159627,
    214.6525378,
    59.86017137,
]
ax.scatter(x, y, s=20, color='purple',alpha=0.5, label='LLM Decode Kernels')
ax.scatter(x_prefill, y_prefill, s=20, color='red',alpha=0.5, label='LLM Prefill Kernels')
# for xi, yi in zip(x, y):
#     ax.text(xi*1.1, yi*1.1, f'{yi:.0f}', fontsize=7)

ax.set_xlabel('Operational Intensity (FLOP/byte)', fontsize=8)
ax.set_ylabel('Performance (TFLOP/s)', fontsize=8)
ax.set_title('Roofline: A100 GPU', fontsize=9)
ax.grid(True, which='both', ls=':', lw=0.4)
ax.legend(fontsize=7, loc='lower right', frameon=False)
plt.tight_layout(pad=0.2)
plt.savefig('roofline_a100_small.pdf', dpi=300)
plt.show()