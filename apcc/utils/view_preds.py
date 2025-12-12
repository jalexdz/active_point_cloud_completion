import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_pcd_xyz(path: str) -> np.ndarray:
    pts = []
    with open(path, "r") as f:
        data_section = False
        for line in f:
            line = line.strip()
            if not data_section:
                if line.startswith("DATA"):
                    data_section = True
                continue
            if line == "":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            pts.append([x, y, z])
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def set_axes_equal(ax, pts_list):
    all_pts = np.concatenate(
        [p for p in pts_list if p is not None and p.shape[0] > 0], axis=0
    )
    if all_pts.shape[0] == 0:
        return

    x_min, y_min, z_min = all_pts.min(axis=0)
    x_max, y_max, z_max = all_pts.max(axis=0)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def plot_object_grid(out_dir: str, object_idx: int, view_indices):
    T = len(view_indices)
    cols = T + 1

    all_pts = []

    inputs = []
    preds = []

    for t, v in enumerate(view_indices):
        in_path = os.path.join(out_dir, f"object_{object_idx}_input_view{v}.pcd")
        pred_path = os.path.join(
            out_dir, f"object_{object_idx}_t{t}_views_{len(view_indices)}.pcd"
        )

        if not os.path.exists(in_path):
            print(f"[WARN] Missing input PCD: {in_path}")
            in_pts = np.zeros((0, 3), dtype=np.float32)
        else:
            in_pts = load_pcd_xyz(in_path)

        if not os.path.exists(pred_path):
            print(f"[WARN] Missing pred PCD: {pred_path}")
            pred_pts = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_pts = load_pcd_xyz(pred_path)

        inputs.append(in_pts)
        preds.append(pred_pts)
        all_pts.append(in_pts)
        all_pts.append(pred_pts)

    # GT
    gt_path = os.path.join(out_dir, f"object_{object_idx}_gt.pcd")
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing GT PCD: {gt_path}")
        gt_pts = np.zeros((0, 3), dtype=np.float32)
    else:
        gt_pts = load_pcd_xyz(gt_path)
    all_pts.append(gt_pts)

    # Create figure
    fig = plt.figure(figsize=(3 * cols, 6))
    axs = []

    for row in range(2):
        row_axes = []
        for col in range(cols):
            ax = fig.add_subplot(2, cols, row * cols + col + 1, projection="3d")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            row_axes.append(ax)
        axs.append(row_axes)

    for t, v in enumerate(view_indices):
        in_pts = inputs[t]
        pred_pts = preds[t]

        ax_in = axs[0][t]
        ax_pred = axs[1][t]

        if in_pts.shape[0] > 0:
            ax_in.scatter(in_pts[:, 0], in_pts[:, 1], in_pts[:, 2], s=1)
        if pred_pts.shape[0] > 0:
            ax_pred.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], s=1)

        ax_in.set_title(f"View {v}", fontsize=8)
        ax_pred.set_title(f"Pred t={t}", fontsize=8)

    ax_gt_top = axs[0][cols - 1]
    ax_gt_bot = axs[1][cols - 1]

    ax_gt_top.set_axis_off()

    if gt_pts.shape[0] > 0:
        ax_gt_bot.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], s=1)
    ax_gt_bot.set_title("GT complete", fontsize=8)

    for row in range(2):
        for col in range(cols):
            set_axes_equal(axs[row][col], all_pts)

    fig.suptitle(f"Object {object_idx}: top=input, bottom=prediction", fontsize=12)
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"object_{object_idx}_grid.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid visualization to {png_path}")



def parse_view_indices(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory with PCDs from infer.py")
    parser.add_argument("--object_idx", type=int, default=0,
                        help="Starting object index")
    parser.add_argument("--num_objects", type=int, default=1,
                        help="How many consecutive objects to visualize")
    parser.add_argument("--views", type=str, default="0,5,10,15",
                        help="Comma-separated view indices (must match infer.py)")

    args = parser.parse_args()
    view_indices = parse_view_indices(args.views)

    for obj_idx in range(args.object_idx, args.object_idx + args.num_objects):
        print(f"\n=== Building grid for object {obj_idx} ===")
        plot_object_grid(args.out_dir, obj_idx, view_indices)
