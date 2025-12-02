import torch

from apcc.cfg import load_cfg
from apcc.data.dataset import MVPSequenceDataset
from apcc.models.apcc_model import APCCModel
from apcc.training.training import sample_queries_and_occupancy


def main():
    print("* Running APCC test...")

    cfg = load_cfg("configs/train.yaml")

    # ---- Dataset ----
    dataset = MVPSequenceDataset(
        prefix="train",
        num_views=cfg.data.num_views,
        random_order=True,
        data_root=cfg.data.root,
        views_per_object=cfg.data.views_per_object,
    )

    print(f"Loaded dataset with {len(dataset)} objects")

    # Grab 1 sample
    label, seq_partials, gt_complete = dataset[0]

    print("seq_partials:", seq_partials.shape)  # [T, N, 3]
    print("gt_complete:", gt_complete.shape)    # [N_gt, 3]
    print("label:", label)

    # ---- Prepare batch ----
    seq_partials = seq_partials.unsqueeze(0)  # [1, T, N, 3]
    gt_complete = gt_complete.unsqueeze(0)    # [1, N_gt, 3]

    B, T, N, _ = seq_partials.shape

    # ---- Query sampling ----
    query_xyz, gt_occ = sample_queries_and_occupancy(
        gt_complete, num_queries=cfg.data.num_queries
    )

    print("query_xyz:", query_xyz.shape)  # [B, Nq, 3]
    print("gt_occ:", gt_occ.shape)        # [B, Nq, 1]

    # ---- Build model ----
    model = APCCModel(cfg.model)

    # Move to CUDA?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    seq_partials = seq_partials.to(device)
    query_xyz = query_xyz.to(device)

    # ---- Run one full sequence ----
    h_prev = None
    for t in range(T):
        pc_t = seq_partials[:, t, :, :]  # [B, N, 3]

        occ_logits, h_prev = model(pc_t, query_xyz, h_prev)

        print(f"t={t}: occ_logits={occ_logits.shape}, h_prev={h_prev.shape}")

    print("test passed")


if __name__ == "__main__":
    main()
