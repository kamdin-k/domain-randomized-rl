import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200).cumsum()

    Path("results").mkdir(exist_ok=True)
    plt.plot(y)
    plt.title("Smoke Test: NumPy + Matplotlib")
    out = Path("results") / "smoke_test.png"
    plt.savefig(out, dpi=150)
    print(f"✅ Saved plot to: {out}")

if __name__ == "__main__":
    main()
