import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- Configuration ---
timestep = 80
prefix = f"mpi_output_t{timestep}"
output_file = f"field_t{timestep}.png"

# --- Load all rank data files ---
files = sorted(glob.glob(f"{prefix}_rank*.dat"))
if not files:
    raise FileNotFoundError(f"❌ No files found with prefix '{prefix}_rank*.dat'")

print(f"📂 Visualizing timestep: {prefix} from {len(files)} MPI ranks")

data_list = []
for f in files:
    try:
        data = np.loadtxt(f)
        if data.size > 0:
            data_list.append(data)
        else:
            print(f"⚠️ Warning: File {f} is empty")
    except Exception as e:
        print(f"❌ Could not load {f}: {e}")

# --- Combine all rank data ---
all_data = np.vstack(data_list)

# --- Extract and sort unique x and y values for reshaping ---
x = all_data[:, 0]
y = all_data[:, 1]
rho = all_data[:, 2]
vx = all_data[:, 3]
vy = all_data[:, 4]

x_unique = np.unique(x)
y_unique = np.unique(y)

nx, ny = len(x_unique), len(y_unique)
if len(x) != nx * ny:
    raise ValueError(f"❌ Cannot reshape: {len(x)} values but grid is {nx}x{ny} = {nx*ny}")

# --- Create grid for plotting ---
x_grid = x.reshape(ny, nx)
y_grid = y.reshape(ny, nx)
rho_grid = rho.reshape(ny, nx)
vx_grid = vx.reshape(ny, nx)
vy_grid = vy.reshape(ny, nx)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Density
im1 = ax1.contourf(x_grid, y_grid, rho_grid, cmap="viridis")
ax1.set_title("Density Field")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
plt.colorbar(im1, ax=ax1)

# Velocity
ax2.quiver(x_grid, y_grid, vx_grid, vy_grid)
ax2.set_title("Velocity Field")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(output_file)
print(f"✅ Saved structured field visualization: {output_file}")
