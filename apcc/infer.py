import os
import argparse
import h5py
import numpy as np
import torch

from apcc.cfg import load_cfg
from apcc.models.apcc_model import APCCModel


def write_pcd(points: np.ndarray, path: str):
    """
    Write points [N, 3] to an ASCII PCD file.
    """
    out_dir = os.path.dirname(path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    N = points.shape[0]

    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {N}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {N}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def occupancy_from_complete(gt_complete: torch.Tensor,
                            query_xyz: torch.Tensor,
                            radius: float = 0.03) -> torch.Tensor:
    """
    Compute GT occupancy for query points given GT complete surface.

    Args:
      gt_complete: [1, N_gt, 3]
      query_xyz:   [1, Nq, 3]
      radius:      threshold distance to consider 'occupied'

    Returns:
      occ_gt: [1, Nq, 1] (1 if query is near GT surface, else 0)
    """
    B, N_gt, _ = gt_complete.shape
    _, Nq, _ = query_xyz.shape
    assert B == 1, "This helper is written for batch size 1."

    pc = gt_complete  # [1, N_gt, 3]
    pc_exp = pc.unsqueeze(2)              # [1, N_gt, 1, 3]
    q_exp = query_xyz.unsqueeze(1)        # [1, 1, Nq, 3]
    diff = pc_exp - q_exp                 # [1, N_gt, Nq, 3]
    dist2 = (diff ** 2).sum(dim=-1)       # [1, N_gt, Nq]
    min_dist2, _ = dist2.min(dim=1)       # [1, Nq]

    occ = (min_dist2 <= radius ** 2).float().unsqueeze(-1)  # [1, Nq, 1]
    return occ


def make_query_grid(gt_complete: torch.Tensor, res: int = 32, padding: float = 0.1):
    """
    Build a regular 3D grid of query points around the GT complete object.

    Args:
      gt_complete: [1, N, 3] (normalized)
      res:         number of steps per axis
      padding:     padding factor around the GT bounding box

    Returns:
      query_xyz: [1, M, 3] with M = res^3
    """
    xyz_min = gt_complete.min(dim=1, keepdim=True).values  # [1, 1, 3]
    xyz_max = gt_complete.max(dim=1, keepdim=True).values  # [1, 1, 3]

    box_min = (xyz_min - padding * (xyz_max - xyz_min)).squeeze(0).squeeze(0)  # [3]
    box_max = (xyz_max + padding * (xyz_max - xyz_min)).squeeze(0).squeeze(0)  # [3]

    xs = torch.linspace(box_min[0], box_max[0], res, device=gt_complete.device)
    ys = torch.linspace(box_min[1], box_max[1], res, device=gt_complete.device)
    zs = torch.linspace(box_min[2], box_max[2], res, device=gt_complete.device)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid = torch.stack([X, Y, Z], dim=-1)  # [res, res, res, 3]
    grid = grid.view(1, -1, 3)            # [1, M, 3]

    return grid


def load_mvp_object(data_root: str, split: str, object_idx: int):
    """
    Loads all 26 partial views and the GT complete for a single object from MVP.

    Returns:
      partials: [26, N, 3] (numpy)
      complete: [N_gt, 3] (numpy)
    """
    if split == "train":
        filename = "MVP_Train_CP.h5"
    elif split == "val":
        filename = "MVP_Test_CP.h5"
    else:
        raise ValueError("split must be 'train' or 'val'")

    path = os.path.join(data_root, filename)
    f = h5py.File(path, "r")

    partials_all = np.array(f["incomplete_pcds"][()])   # [62400, 2048, 3]
    complete_all = np.array(f["complete_pcds"][()])     # [2400, 2048, 3]
    f.close()

    views_per_object = 26
    start = object_idx * views_per_object
    end = start + views_per_object

    partials = partials_all[start:end]       # [26, N, 3]
    complete = complete_all[object_idx]      # [N_gt, 3]

    return partials, complete


def normalize_object(partials: np.ndarray, complete: np.ndarray):
    """
    Normalize object coordinates using the GT complete cloud.

    Returns:
      partials_norm: [T, N, 3]
      complete_norm: [N_gt, 3]
      center: [3]
      scale: scalar
    """
    center = complete.mean(axis=0, keepdims=True)             # [1, 3]
    dists = np.linalg.norm(complete - center, axis=1)
    scale = np.max(dists) + 1e-9

    partials_norm = (partials - center) / scale
    complete_norm = (complete - center) / scale

    return partials_norm, complete_norm, center.squeeze(0), scale


def run_sequence_inference(cfg_path: str,
                           ckpt_path: str,
                           data_root: str,
                           split: str,
                           object_idx: int,
                           view_indices,
                           out_dir: str,
                           device_str: str = "cuda",
                           grid_res: int = 32,
                           occ_thresh: float = 0.5):
    """
    Run sequential inference for a single object over a specified set of view indices.
    Saves:
      - input partials (denormalized) as PCD
      - GT complete as PCD
      - per-timestep predictions as PCD
    Prints:
      - per-timestep occupancy accuracy on the query grid.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(cfg_path)

    # --- Load model + checkpoint ---
    model = APCCModel(cfg.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Load MVP object ---
    partials_all, complete = load_mvp_object(data_root, split, object_idx)
    partials_all = partials_all.astype(np.float32)
    complete = complete.astype(np.float32)

    # --- Normalize ---
    partials_norm, complete_norm, center, scale = normalize_object(partials_all, complete)

    # pick the views we want: view_indices is a list like [0, 5, 10, 15]
    partials_seq = partials_norm[view_indices]  # [T, N, 3]

    # --- Save the chosen input partials (denormalized) for visualization ---
    for i, v_idx in enumerate(view_indices):
        partial_norm = partials_seq[i]                    # [N, 3], normalized
        partial_world = partial_norm * scale + center     # [N, 3], numpy

        in_path = os.path.join(
            out_dir, f"object_{object_idx}_input_view{v_idx}.pcd"
        )
        write_pcd(partial_world, in_path)
        print(f"Saved input partial for view {v_idx} to {in_path} "
              f"({partial_world.shape[0]} points)")

    # convert to torch
    partials_seq_t = torch.from_numpy(partials_seq).to(device)     # [T, N, 3]
    complete_t = torch.from_numpy(complete_norm).unsqueeze(0).to(device)  # [1, N_gt, 3]

    # --- Make query grid once ---
    query_xyz = make_query_grid(complete_t, res=grid_res, padding=0.1)  # [1, M, 3]

    # --- GT occupancy on the same grid for accuracy metrics ---
    gt_occ_grid = occupancy_from_complete(complete_t, query_xyz)  # [1, M, 1]

    # --- Save GT once (denormalized) ---
    gt_world = complete_norm * scale + center  # numpy [N_gt, 3]
    gt_path = os.path.join(out_dir, f"object_{object_idx}_gt.pcd")
    write_pcd(gt_world, gt_path)
    print(f"Saved GT complete cloud for object {object_idx} to {gt_path} "
          f"({gt_world.shape[0]} points)")

    # --- Roll through sequence ---
    h_prev = None
    T = partials_seq_t.shape[0]
    B = 1

    with torch.no_grad():
        for t in range(T):
            pc_t = partials_seq_t[t].unsqueeze(0)  # [1, N, 3]

            occ_logits, h_prev = model(pc_t, query_xyz, h_prev)  # [1, M, 1]
            probs = torch.sigmoid(occ_logits)                    # [1, M, 1]

            # ---- Accuracy vs GT occupancy on the grid ----
            preds = (probs > occ_thresh).float()                 # [1, M, 1]
            acc_t = (preds == gt_occ_grid).float().mean().item()
            print(f"[object {object_idx}] t = {t}, grid occupancy accuracy = {acc_t:.4f}")

            # ---- Extract predicted occupied points for visualization ----
            occ_probs = probs.view(B, -1)                        # [1, M]
            mask = occ_probs[0] > occ_thresh
            pred_points_norm = query_xyz[0][mask]                # [K, 3]

            pred_points_norm_np = pred_points_norm.cpu().numpy()
            pred_points_world = pred_points_norm_np * scale + center  # [K, 3]

            out_path = os.path.join(
                out_dir, f"object_{object_idx}_t{t}_views_{len(view_indices)}.pcd"
            )
            write_pcd(pred_points_world, out_path)
            print(f"Saved timestep {t} prediction to {out_path} "
                  f"({pred_points_world.shape[0]} points)")


def parse_view_indices(s: str):
    # e.g. "0,5,10,15" -> [0, 5, 10, 15]
    return [int(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/infer.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pth")
    parser.add_argument("--data_root", type=str, default="/data",
                        help="Root with MVP_Train_CP.h5 / MVP_Test_CP.h5")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--object_idx",
        type=int,
        default=0,
        help="Starting object index"
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=1,
        help="How many consecutive objects to run"
    )
    parser.add_argument(
        "--views",
        type=str,
        default="0,5,10,15",
        help="Comma-separated view indices from [0..25]"
    )
    parser.add_argument("--out_dir", type=str, default="outputs/infer_vis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid_res", type=int, default=32)
    parser.add_argument("--occ_thresh", type=float, default=0.5)

    args = parser.parse_args()
    view_indices = parse_view_indices(args.views)

    for obj_idx in range(args.object_idx, args.object_idx + args.num_objects):
        print(f"\n=== Running inference for object {obj_idx} ===")
        run_sequence_inference(
            cfg_path=args.cfg,
            ckpt_path=args.ckpt,
            data_root=args.data_root,
            split=args.split,
            object_idx=obj_idx,
            view_indices=view_indices,
            out_dir=args.out_dir,
            device_str=args.device,
            grid_res=args.grid_res,
            occ_thresh=args.occ_thresh,
        )
