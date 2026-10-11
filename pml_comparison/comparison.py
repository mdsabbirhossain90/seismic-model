import matplotlib.pyplot as plt
from PIL import Image


pml20 = [
    "../pml_model/results/PML20_a0.3km/wavefield_t2.00s.png",
    "../pml_model/results/PML20_a0.3km/wavefield_t4.00s.png",
    "../pml_model/results/PML20_a0.3km/wavefield_t6.00s.png",
    "../pml_model/results/PML20_a0.3km/wavefield_t8.00s.png"
]


pml30 = [
    "../pml_model/results/PML30_a0.3km/wavefield_t2.00s.png",
    "../pml_model/results/PML30_a0.3km/wavefield_t4.00s.png",
    "../pml_model/results/PML30_a0.3km/wavefield_t6.00s.png",
    "../pml_model/results/PML30_a0.3km/wavefield_t8.00s.png"
]


pml40 = [
    "../pml_model/results/PML40_a0.3km/wavefield_t2.00s.png",
    "../pml_model/results/PML40_a0.3km/wavefield_t4.00s.png",
    "../pml_model/results/PML40_a0.3km/wavefield_t6.00s.png",
    "../pml_model/results/PML40_a0.3km/wavefield_t8.00s.png"
]


fig, axes = plt.subplots(
    4,
    3,
    figsize=(15,18)
)


fig.suptitle(
    "Wavefield Comparison (PML 20 vs PML 30 vs PML 40)",
    fontsize=22,
    fontweight="bold"
)


groups = [
    ("PML 20", pml20),
    ("PML 30", pml30),
    ("PML 40", pml40)
]


times = [
    "t = 2 s",
    "t = 4 s",
    "t = 6 s",
    "t = 8 s"
]


for col,(title,files) in enumerate(groups):

    axes[0,col].set_title(
        title,
        fontsize=18,
        fontweight="bold"
    )

    for row,file in enumerate(files):

        img = Image.open(file)

        axes[row,col].imshow(img)
        axes[row,col].axis("off")

        if col == 0:
            axes[row,col].set_ylabel(
                times[row],
                fontsize=14
            )


plt.tight_layout()

plt.savefig(
    "output/Wavefield_PML20_PML30_PML40.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "output/Wavefield_PML20_PML30_PML40.pdf",
    bbox_inches="tight"
)

plt.show()