import random
import math
import matplotlib.pyplot as plt



def sample(N, func, H, W):
    if func == "random":
        return [(random.randint(0, W-8), random.randint(0, H-4)) for _ in range(N)]
    
    elif func == "gradient":
        return [(int((i / N) * (W-8)), int((i / N) * (H-4))) for i in range(N)]
    
    elif func == "periodic":
        period = max(1, N // 10)
        return [( (i % period) * W // period, (i % period) * H // period ) for i in range(N)]
    
    elif func == "blurred":
        center_x, center_y = W // 2, H // 2
        return [(int(random.gauss(center_x, W//8)), int(random.gauss(center_y, H//8))) 
                for _ in range(N)]
    
    elif func == "skew":
        return [(int((random.random() ** 2) * (W-8)), int((random.random() ** 0.5) * (H-4))) 
                for _ in range(N)]

    
    else:
        raise ValueError(f"Unknown pattern: {func}")





patterns = ["random", "gradient", "periodic", "blurred", "skew"]
W, H = 64, 64
N = 200

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, pattern in enumerate(patterns):
    points = sample(N, pattern, W, H)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    axes[i].scatter(x_vals, y_vals, alpha=0.6, s=10)
    axes[i].set_title(f"{pattern.capitalize()} Pattern")
    axes[i].set_xlim(0, W)
    axes[i].set_ylim(0, H)
    axes[i].invert_yaxis()  # Flip Y for top-down memory view feel
    axes[i].set_aspect('equal')

plt.tight_layout()
plt.show()
