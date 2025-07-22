import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import subprocess

# --- Config ---
timesteps = range(0, 201, 20)  # frames for t = 0, 20, ..., 200
frame_dir = "frames"
video_output = "lid_driven_cavity.mp4"

os.makedirs(frame_dir, exist_ok=True)

for timestep in timesteps:
    prefix = f"mpi_output_t{timestep}"
    output_file = os.path.join(frame_dir, f"field_t{timestep:03d}.png")
    files = sorted(glob.glob(f"{prefix}_rank*.dat"))

    if not files:
        print(f"❌ No files found for timestep {timestep}")
        continue

    print(f"📂 Loading: {prefix} from {len(files)} MPI ranks")
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

    if not data_list:
        print(f"⚠️ No valid data loaded for timestep {timestep}")
        continue

    all_data = np.vstack(data_list)

    x = all_data[:, 0]
    y = all_data[:, 1]
    rho = all_data[:, 2]
    vx = all_data[:, 3]
    vy = all_data[:, 4]

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)

    if len(x) != nx * ny:
        print(f"❌ Grid shape mismatch at timestep {timestep}")
        continue

    x_grid = x.reshape(ny, nx)
    y_grid = y.reshape(ny, nx)
    rho_grid = rho.reshape(ny, nx)
    vx_grid = vx.reshape(ny, nx)
    vy_grid = vy.reshape(ny, nx)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.contourf(x_grid, y_grid, rho_grid, cmap="viridis")
    ax1.set_title(f"Density Field (t = {timestep})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1)

    ax2.quiver(x_grid, y_grid, vx_grid, vy_grid)
    ax2.set_title(f"Velocity Field (t = {timestep})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"✅ Saved: {output_file}")

# --- Create video using ffmpeg ---
print("🎥 Compiling video...")

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-framerate", "5",
    "-pattern_type", "glob",
    "-i", os.path.join(frame_dir, "field_t*.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    video_output
]

try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ Video created: {video_output}")
except Exception as e:
    print(f"❌ Failed to create video: {e}")
