"""
Generic training loop for the neural Royston-Parmar variants.

Deliberately variant-agnostic: it only needs a model exposing
`loss(log_t, x_cont, event_type, x_cat)`, `param_groups()`,
`min_spline_deriv()` and `checkpoint_dict()`, so later variants reuse it
unchanged.

Logs per epoch: train loss, validation loss, min s'_k per cause, and
gradient norm per parameter group - the last two are the diagnostics for
the failure modes this design is guarding against (silent non-monotonicity
and the spline coefficients learning far more slowly than the encoder).
"""

from dataclasses import dataclass, field

import torch


@dataclass
class TrainConfig:
    max_epochs: int = 200
    batch_size: int = 512
    patience: int = 15                 # early stopping on validation loss
    scheduler: str = "cosine"          # "cosine" | "plateau" | "none"
    weight_decay: float = 0.0
    grad_clip: float = None
    seed: int = 42
    log_every: int = 10
    verbose: bool = True


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    min_deriv: dict
    grad_norms: dict
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _grad_norms(param_groups) -> dict:
    out = {}
    for g in param_groups:
        total = 0.0
        for p in g["params"]:
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        out[g.get("name", "group")] = total ** 0.5
    return out


def train_model(model,
                train_batch: dict,
                val_batch: dict,
                config: TrainConfig = None,
                checkpoint_path: str = None):
    """Fit `model`; returns (history, best_checkpoint).

    train_batch / val_batch are dicts with keys `log_t`, `x_cont`,
    `event_type`, and optionally `x_cat`. Full-batch when
    `batch_size >= n`, otherwise shuffled minibatches.

    Early stopping restores the best-validation weights before returning,
    and the returned checkpoint carries the basis and scaler with it.
    """
    config = config or TrainConfig()
    set_seed(config.seed)

    groups = model.param_groups()
    opt = torch.optim.Adam(groups, weight_decay=config.weight_decay)
    if config.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config.max_epochs)
    elif config.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=max(1, config.patience // 3), factor=0.5)
    else:
        sched = None

    n = train_batch["log_t"].numel()
    generator = torch.Generator().manual_seed(config.seed)

    history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(config.max_epochs):
        model.train()
        if config.batch_size >= n:
            order = [torch.arange(n)]
        else:
            perm = torch.randperm(n, generator=generator)
            order = perm.split(config.batch_size)

        epoch_loss, seen = 0.0, 0
        for idx in order:
            opt.zero_grad()
            loss = model.loss(
                train_batch["log_t"][idx], train_batch["x_cont"][idx],
                train_batch["event_type"][idx],
                None if train_batch.get("x_cat") is None
                else train_batch["x_cat"][idx],
            )
            loss.backward()
            gnorms = _grad_norms(groups)
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            epoch_loss += loss.item() * idx.numel()
            seen += idx.numel()

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(
                val_batch["log_t"], val_batch["x_cont"],
                val_batch["event_type"], val_batch.get("x_cat"),
            ).item()

        if sched is not None:
            sched.step(val_loss) if config.scheduler == "plateau" else sched.step()

        min_deriv = model.min_spline_deriv()
        # Variant-specific diagnostics, e.g. the distribution of b_k(z) across
        # the batch for Variant B: a b_k collapsing toward 0 or drifting large
        # is the signature of an identifiability problem surviving.
        extra = (model.batch_diagnostics(val_batch)
                 if hasattr(model, "batch_diagnostics") else {})
        log = EpochLog(epoch, epoch_loss / seen, val_loss, min_deriv, gnorms, extra)
        history.append(log)
        if config.verbose and (epoch % config.log_every == 0
                               or epoch == config.max_epochs - 1):
            derivs = ", ".join(f"s'_{k}={v:+.4f}" for k, v in min_deriv.items())
            norms = ", ".join(f"|g|_{k}={v:.2e}" for k, v in gnorms.items())
            extras = "  " + " ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
            print(f"  epoch {epoch:4d}  train {log.train_loss:.6f}  "
                  f"val {val_loss:.6f}  {derivs}  {norms}{extras}")
        if any(v <= 0 for v in min_deriv.values()):
            print(f"  *** WARNING epoch {epoch}: non-positive spline derivative "
                  f"{min_deriv} - cumulative hazard is not monotone ***")

        if val_loss < best_val - 1e-9:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                if config.verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best val {best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = model.checkpoint_dict()
    ckpt["best_val_loss"] = best_val
    ckpt["epochs_run"] = len(history)
    if checkpoint_path:
        torch.save(ckpt, checkpoint_path)
    return history, ckpt
