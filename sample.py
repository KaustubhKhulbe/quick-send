import random
import math
import matplotlib.pyplot as plt
import numpy as np


def sample(N, func, H, W):
    if func == "random":
        return [(random.randint(0, W - 8), random.randint(0, H - 4)) for _ in range(N)]

    elif func == "periodic":
        stride = 3
        points = []

        for i in range(0, H, 10):
            for j in range(0, W, 2):
                new_x = j
                new_y = (j % stride) + (i - (i % stride)) 
                points.append((new_x, new_y))

        return points[:N]
        
    elif func == "strided":
        rows = int(math.sqrt(N * H / W))  # Maintain aspect ratio
        cols = max(1, N // rows)
        xs = [int((i / (cols - 1)) * (W - 8)) for i in range(cols)]
        ys = [int((j / (rows - 1)) * (H - 4)) for j in range(rows)]
        points = [(x, y) for y in ys for x in xs]
        return points[:N]  # Trim to N if oversampled

    elif func == "blurred":
        center_x1, center_y1 = W // 4, H // 4
        center_x2, center_y1 = W // (4/3), H // 4
        center_x1, center_y2 = W // 4, H // (4/3)
        center_x2, center_y2 = W // (4/3), H // (4/3)

        def helper(c1, c2, num):
            return [(int(random.gauss(c1, W // 10)), int(random.gauss(c2, H // 10)))
                for _ in range(num)]

        a = np.array(helper(center_x1, center_y1, N // 4))
        b = np.array(helper(center_x2, center_y1, N // 4))
        c = np.array(helper(center_x1, center_y2, N // 4))
        d = np.array(helper(center_x2, center_y2, N // 4))
        return np.concatenate((a,b,c,d))

    elif func == "skew":
        return [(int((random.random() ** 2) * (W - 8)), int((random.random() ** 0.5) * (H - 4)))
                for _ in range(N)]

    elif func == "normal":
        points = [(x, y) for y in range(H) for x in range(W)] 
        return points[:N]

    else:
        raise ValueError(f"Unknown pattern: {func}")





patterns = ["random", "periodic", "strided", "blurred", "skew", "normal"]
W, H = 64, 64
N = 500

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
