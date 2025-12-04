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
            visible_mask = compute_visibility_mask(seq_partials[:, t, :, :], query_xyz)  # [B, Nq]

            # Flatten logits and targets
            logits_flat = occ_logits.view(B, -1)
            gt_flat = gt_occ.view(B, -1)

            # If no visible queries, skip this timestep (avoid NaNs)
            # if visible_mask.any():
            #     loss_t = criterion(
            #         logits_flat[visible_mask],
            #         gt_flat[visible_mask],
            #     )
            loss_t = criterion(
                logits_flat,
                gt_flat,
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

def compute_visibility_mask(pc_t: torch.Tensor,
                            query_xyz: torch.Tensor,
                            radius: float = 0.03) -> torch.Tensor:
    """
    pc_t:      [B, N, 3]   partial cloud at timestep t (normalized coords)
    query_xyz: [B, Nq, 3]  query points for occupancy

    Returns:
      visible_mask: [B, Nq] boolean
        visible_mask[b, j] = True if query j is within 'radius'
        of *any* point in pc_t[b].
    """
    # Ensure shapes
    assert pc_t.dim() == 3 and query_xyz.dim() == 3
    B, N, _ = pc_t.shape
    _, Nq, _ = query_xyz.shape

    # Expand to pairwise distances
    # pc_exp: [B, N, 1, 3]
    # q_exp:  [B, 1, Nq, 3]
    pc_exp = pc_t.unsqueeze(2)
    q_exp = query_xyz.unsqueeze(1)

    # [B, N, Nq]
    diff = pc_exp - q_exp
    dist2 = (diff ** 2).sum(dim=-1)

    # For each query, get distance to closest partial point
    # min_dist2: [B, Nq]
    min_dist2, _ = dist2.min(dim=1)

    visible_mask = (min_dist2 <= radius ** 2)  # [B, Nq] bool
    return visible_mask

def validate_one_epoch(model: torch.nn.Module,
                       dataloader: DataLoader,
                       device: torch.device,
                       epoch: int,
                       writer: Optional[SummaryWriter] = None,
                       num_queries: int = 512):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            labels, seq_partials, gt_complete, center, scale = batch
            seq_partials = seq_partials.to(device).float()
            gt_complete = gt_complete.to(device).float()

            B, T, N, _ = seq_partials.shape

            query_xyz, gt_occ = sample_queries_and_occupancy(
                gt_complete, num_queries=num_queries
            )
            query_xyz = query_xyz.to(device)
            gt_occ = gt_occ.to(device)

            h_prev = None
            total_loss = 0.0

            for t in range(T):
                pc_t = seq_partials[:, t, :, :]  # [B, N, 3]
                occ_logits, h_prev = model(pc_t, query_xyz, h_prev)

                visible_mask = compute_visibility_mask(seq_partials[:, t, :, :], query_xyz)  # [B, Nq]

                # Flatten logits and targets
                logits_flat = occ_logits.view(B, -1)
                gt_flat = gt_occ.view(B, -1)

                # If no visible queries, skip this timestep (avoid NaNs)
                # if visible_mask.any():
                #     loss_t = criterion(
                #         logits_flat[visible_mask],
                #         gt_flat[visible_mask],
                #     )
                
                loss_t = criterion(
                    logits_flat,
                    gt_flat,
                )  
                total_loss += loss_t   
   

            total_loss = total_loss / T
            running_loss += total_loss.item()

    epoch_loss = running_loss / num_batches
    if writer is not None:
        writer.add_scalar("val/epoch_loss", epoch_loss, epoch)

    print(f"[Val] Epoch {epoch} avg loss: {epoch_loss:.4f}")
    return epoch_loss


def train_model(model: torch.nn.Module,
                train_dataset,
                val_dataset,
                cfg,
                device: torch.device):
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    writer = SummaryWriter(log_dir=cfg.logging.logdir)

    best_val_loss = float("inf")

    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            log_every=cfg.logging.print_every,
            num_queries=cfg.data.num_queries if hasattr(cfg.data, "num_queries") else 512,
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            device,
            epoch,
            writer,
            num_queries=cfg.data.num_queries if hasattr(cfg.data, "num_queries") else 512,
        )

        # save best on validation
        if val_loss < best_val_loss and cfg.logging.save_last:
            best_val_loss = val_loss
            ckpt_path = f"{cfg.logging.ckpt_dir}/best.pth"
            print(f"Saving new best model to {ckpt_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

    writer.close()
