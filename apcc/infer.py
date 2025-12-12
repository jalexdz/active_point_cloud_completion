import os
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from apcc.cfg import load_cfg
from apcc.models.apcc_model import APCCModel


def write_pcd(points: np.ndarray, path: str):
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


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")

    # [Nx, Ny, 3]
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist2 = (diff ** 2).sum(dim=-1)  # [Nx, Ny]

    min_x, _ = dist2.min(dim=1)  # [Nx]
    min_y, _ = dist2.min(dim=0)  # [Ny]

    return (min_x.mean() + min_y.mean()).item()


def occupancy_from_complete(gt_complete: torch.Tensor,
                            query_xyz: torch.Tensor,
                            radius: float = 0.03) -> torch.Tensor:
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
    gt_occ_flat = gt_occ_grid.view(-1)                            # [M]
    gt_occ_bool = gt_occ_flat > 0.5

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

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    iou_list = []
    chamfer_list = []

    with torch.no_grad():
        for t in range(T):
            pc_t = partials_seq_t[t].unsqueeze(0)  # [1, N, 3]

            occ_logits, h_prev = model(pc_t, query_xyz, h_prev)  # [1, M, 1]
            probs = torch.sigmoid(occ_logits)                    # [1, M, 1]

            preds = (probs > occ_thresh).float()                 # [1, M, 1]
            preds_flat = preds.view(-1)                          # [M]
            preds_bool = preds_flat > 0.5

            # ---- confusion counts ----
            tp = ((preds_bool == 1) & (gt_occ_bool == 1)).sum().item()
            tn = ((preds_bool == 0) & (gt_occ_bool == 0)).sum().item()
            fp = ((preds_bool == 1) & (gt_occ_bool == 0)).sum().item()
            fn = ((preds_bool == 0) & (gt_occ_bool == 1)).sum().item()

            total = tp + tn + fp + fn
            acc_t = (tp + tn) / total if total > 0 else 0.0

            prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_t  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec_t + rec_t > 0:
                f1_t = 2 * prec_t * rec_t / (prec_t + rec_t)
            else:
                f1_t = 0.0

            denom_iou = tp + fp + fn
            iou_t = tp / denom_iou if denom_iou > 0 else 0.0

            acc_list.append(acc_t)
            prec_list.append(prec_t)
            rec_list.append(rec_t)
            f1_list.append(f1_t)
            iou_list.append(iou_t)

            # ---- Chamfer distance vs GT complete cloud ----
            occ_probs = probs.view(B, -1)             # [1, M]
            mask = occ_probs[0] > occ_thresh
            pred_points_norm = query_xyz[0][mask]     # [K, 3]

            gt_points_norm = complete_t[0]            # [N_gt, 3]

            cd_t = chamfer_distance(
                pred_points_norm,
                gt_points_norm
            )
            chamfer_list.append(cd_t)

            print(
                f"[object {object_idx}] t = {t}, "
                f"acc={acc_t:.4f}, prec={prec_t:.4f}, rec={rec_t:.4f}, "
                f"f1={f1_t:.4f}, IoU={iou_t:.4f}, CD={cd_t:.6f}"
            )

            # ---- Save predicted occupied points (world coords) ----
            pred_points_norm_np = pred_points_norm.cpu().numpy()
            pred_points_world = pred_points_norm_np * scale + center  # [K, 3]

            out_path = os.path.join(
                out_dir, f"object_{object_idx}_t{t}_views_{len(view_indices)}.pcd"
            )
            write_pcd(pred_points_world, out_path)
            print(f"Saved timestep {t} prediction to {out_path} "
                  f"({pred_points_world.shape[0]} points)")

    # return metrics for this object
    return {
        "acc": np.array(acc_list),
        "prec": np.array(prec_list),
        "rec": np.array(rec_list),
        "f1": np.array(f1_list),
        "iou": np.array(iou_list),
        "cd": np.array(chamfer_list),
        "T": T,
    }


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
    parser.add_argument("--out_dir", type=str, default="outputs/infer_vis_wo_gru")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid_res", type=int, default=32)
    parser.add_argument("--occ_thresh", type=float, default=0.5)

    args = parser.parse_args()
    view_indices = parse_view_indices(args.views)
    os.makedirs(args.out_dir, exist_ok=True)

    metric_sums = None
    num_objects = args.num_objects
    all_metrics = []

    for obj_idx in range(args.object_idx, args.object_idx + num_objects):
        print(f"\n=== Running inference for object {obj_idx} ===")
        metrics = run_sequence_inference(
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
        all_metrics.append(metrics)

        if metric_sums is None:
            metric_sums = {
                k: metrics[k].copy()
                for k in ["acc", "prec", "rec", "f1", "iou", "cd"]
            }
        else:
            for k in metric_sums.keys():
                metric_sums[k] += metrics[k]

    # average over objects
    avg_metrics = {k: v / num_objects for k, v in metric_sums.items()}

    # save for later plotting / comparison (e.g. GRU vs no-GRU)
    np.savez(
        os.path.join(args.out_dir, "metrics_seq.npz"),
        view_indices=np.array(view_indices, dtype=np.int32),
        avg_acc=avg_metrics["acc"],
        avg_prec=avg_metrics["prec"],
        avg_rec=avg_metrics["rec"],
        avg_f1=avg_metrics["f1"],
        avg_iou=avg_metrics["iou"],
        avg_cd=avg_metrics["cd"],
    )

    print("\n=== Average metrics over objects ===")
    for t, v in enumerate(view_indices):
        print(
            f"t={t} (view {v}): "
            f"acc={avg_metrics['acc'][t]:.4f}, "
            f"f1={avg_metrics['f1'][t]:.4f}, "
            f"IoU={avg_metrics['iou'][t]:.4f}, "
            f"CD={avg_metrics['cd'][t]:.6f}"
        )

    # --- Quick plots of how metrics evolve with views ---
    steps = np.arange(len(view_indices))

    plt.figure()
    plt.plot(steps, avg_metrics["acc"], label="Accuracy")
    plt.plot(steps, avg_metrics["f1"],  label="F1")
    plt.plot(steps, avg_metrics["iou"], label="IoU")
    plt.xlabel("Timestep index")
    plt.ylabel("Score")
    plt.xticks(steps, view_indices)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "metrics_curve.png"), dpi=200)

    plt.figure()
    plt.plot(steps, avg_metrics["cd"], label="Chamfer")
    plt.xlabel("Timestep index")
    plt.ylabel("Chamfer distance")
    plt.xticks(steps, view_indices)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "chamfer_curve.png"), dpi=200)

    print(f"\nSaved metrics curves to {args.out_dir}")
