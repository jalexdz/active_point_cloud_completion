# apcc/training/training.py

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def sample_queries_and_occupancy(gt_complete: torch.Tensor,
                                 num_queries: int = 512):
  
    device = gt_complete.device
    B, N_gt, _ = gt_complete.shape

    # --- 1) Positive samples: points from the GT surface ---
    num_pos = num_queries // 2
    num_neg = num_queries - num_pos

    idx = torch.randint(0, N_gt, (B, num_pos), device=device)  # [B, num_pos]
    pos_xyz = gt_complete[torch.arange(B).unsqueeze(-1), idx]  # [B, num_pos, 3]

    # --- 2) Negative samples: random points in bounding box around GT ---
    xyz_min = gt_complete.min(dim=1, keepdim=True).values  # [B, 1, 3]
    xyz_max = gt_complete.max(dim=1, keepdim=True).values  # [B, 1, 3]

    # expand box slightly
    padding = 0.1 * (xyz_max - xyz_min + 1e-6)
    xyz_min = xyz_min - padding
    xyz_max = xyz_max + padding

    neg_xyz = torch.rand(B, num_neg, 3, device=device) * (xyz_max - xyz_min) + xyz_min

    # naive assumption: random box samples are mostly empty
    pos_occ = torch.ones(B, num_pos, 1, device=device)
    neg_occ = torch.zeros(B, num_neg, 1, device=device)

    query_xyz = torch.cat([pos_xyz, neg_xyz], dim=1)   # [B, num_queries, 3]
    gt_occ = torch.cat([pos_occ, neg_occ], dim=1)      # [B, num_queries, 1]

    return query_xyz, gt_occ


def create_dataloader(dataset,
                      batch_size: int,
                      num_workers: int,
                      shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_one_epoch(model: torch.nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    writer: Optional[SummaryWriter] = None,
                    log_every: int = 10,
                    num_queries: int = 512):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    num_batches = len(dataloader)

    start_time = time.time()
    global_step_base = epoch * num_batches

    for batch_idx, batch in enumerate(dataloader):
        # unpack batch from dataset
        # expected: label, seq_partials, gt_complete
        labels, seq_partials, gt_complete, center, scale = batch
        # seq_partials: [B, T, N, 3]
        # gt_complete:  [B, N_gt, 3]

        seq_partials = seq_partials.to(device).float()
        gt_complete = gt_complete.to(device).float()

        B, T, N, _ = seq_partials.shape

        # sample fixed queries per batch (same for all timesteps)
        query_xyz, gt_occ = sample_queries_and_occupancy(
            gt_complete, num_queries=num_queries
        )
        query_xyz = query_xyz.to(device)
        gt_occ = gt_occ.to(device)

        optimizer.zero_grad()

        # roll through the sequence
        h_prev = None
        total_loss = 0.0

        for t in range(T):
            pc_t = seq_partials[:, t, :, :]  # [B, N, 3]

            occ_logits, h_prev = model(pc_t, query_xyz, h_prev)  # [B, Nq, 1]

            # BCE expects [B, Nq] or [B*Nq]; we can flatten
            loss_t = criterion(
                occ_logits.view(B, -1),
                gt_occ.view(B, -1),
            )
            total_loss += loss_t

        # full-sequence supervision: average over timesteps
        total_loss = total_loss / T
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        global_step = global_step_base + batch_idx

        if (batch_idx + 1) % log_every == 0:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch} | Batch {batch_idx+1}/{num_batches}] "
                f"Loss: {total_loss.item():.4f} (avg: {avg_loss:.4f}) | "
                f"Time: {elapsed:.1f}s"
            )
            if writer is not None:
                writer.add_scalar("train/batch_loss", total_loss.item(), global_step)

    epoch_loss = running_loss / num_batches
    if writer is not None:
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

    print(f"Epoch {epoch} done. Avg loss: {epoch_loss:.4f}")
    return epoch_loss


def train_model(model: torch.nn.Module,
                train_dataset,
                cfg,
                device: torch.device):
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    writer = SummaryWriter(log_dir=cfg.logging.logdir)

    best_loss = float("inf")

    for epoch in range(cfg.train.epochs):
        epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            log_every=cfg.logging.print_every,
            num_queries=cfg.data.num_queries if hasattr(cfg.data, "num_queries") else 512,
        )

        # optional: save best
        if epoch_loss < best_loss and cfg.logging.save_last:
            best_loss = epoch_loss
            ckpt_path = f"{cfg.logging.ckpt_dir}/best.pth"
            print(f"Saving new best model to {ckpt_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                ckpt_path,
            )

    writer.close()
