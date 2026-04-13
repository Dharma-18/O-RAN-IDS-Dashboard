# ==============================================================================
# Trainer — Two-Phase Training with Epoch Callbacks
# ==============================================================================
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from typing import Callable, Dict, List, Optional, Tuple
from .model import AttentionGuidedEgoGAT, nt_xent_loss


def train(
    model:            AttentionGuidedEgoGAT,
    ego_dataset:      list,
    device:           torch.device,
    pretrain_epochs:  int = 5,
    finetune_epochs:  int = 10,
    lr:               float = 0.001,
    batch_size:       int = 64,
    temperature:      float = 0.5,
    callback:         Optional[Callable] = None,
) -> Tuple[AttentionGuidedEgoGAT, Dict[str, List]]:
    """
    Phase 1 — Contrastive pre-training (NT-Xent).
        Two forward passes with different dropout masks act as graph augmentations.

    Phase 2 — Supervised fine-tuning (BCE with logits).
        Standard binary classification.

    callback(phase: str, epoch: int, metrics: dict) is called after every epoch.
    Returns (trained_model, history_dict).
    """
    # ---- Filter out -1 label graphs for supervised phase ----
    valid_dataset  = [g for g in ego_dataset if g.y.item() >= 0]
    loader_all     = DataLoader(ego_dataset,   batch_size=batch_size, shuffle=True,  drop_last=True)
    loader_valid   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pretrain_epochs + finetune_epochs, eta_min=lr * 0.1
    )
    criterion  = nn.BCEWithLogitsLoss()

    history: Dict[str, List] = {
        'pretrain_loss': [],
        'finetune_loss': [],
        'finetune_acc':  [],
    }

    model.to(device)

    # ================================================================
    # PHASE 1 — Contrastive Pre-training
    # ================================================================
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        total_loss, num_batches = 0.0, 0

        for batch in loader_all:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Two stochastic forward passes = two augmented views
            _, z1 = model(batch, return_embeds=True)
            _, z2 = model(batch, return_embeds=True)

            loss = nt_xent_loss(z1, z2, temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        history['pretrain_loss'].append(avg_loss)

        if callback:
            callback('pretrain', epoch, {'loss': avg_loss})

    # ================================================================
    # PHASE 2 — Supervised Fine-tuning
    # ================================================================
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader_valid:
            batch  = batch.to(device)
            labels = batch.y.float()

            optimizer.zero_grad()
            logits = model(batch).squeeze(-1)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds       = (torch.sigmoid(logits) > 0.5).long()
            correct    += (preds == batch.y).sum().item()
            total      += len(batch.y)

        scheduler.step()
        avg_loss = total_loss / max(len(loader_valid), 1)
        acc      = correct / max(total, 1)
        history['finetune_loss'].append(avg_loss)
        history['finetune_acc'].append(acc)

        if callback:
            callback('finetune', epoch, {'loss': avg_loss, 'acc': acc})

    return model, history
