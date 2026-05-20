"""Wrapper around `lerobot-train` that runs a validation pass on a held-out dataset.

Adds these flags on top of the normal lerobot-train CLI:
    --val_dataset.repo_id=<repo_id>     (required to enable validation)
    --val_dataset.root=<path>           (optional local root)
    --val_freq=<int>                    (default 1000)
    --val_batches=<int>                 (default 20)
    --val_rollout_batches=<int>         (default 0 = off; runs full action-chunk
                                         sampling and compares to GT chunks)
    --val_rollout_scope=executed|full   (default "executed" — first
                                         n_action_steps of the horizon)
    --val_rollout_plot_samples=<int>    (default 4; per-joint pred-vs-GT plots
                                         logged as wandb images)

Example:
    uv run python robot_learning_2026/scripts/train_with_validation.py \\
        --dataset.repo_id=DerBoroter/full_fold_tristan \\
        --policy.type=multi_task_dit \\
        --val_dataset.repo_id=DerBoroter/full_fold_val \\
        --val_freq=1000 --val_batches=20 \\
        ...

Implementation: monkeypatches `lerobot.scripts.lerobot_train` to capture the
policy + preprocessor + val dataloader, then wraps `update_policy` so that
every `val_freq` steps we run `val_batches` forward passes in eval mode and
log mean loss plus per-component breakdown to wandb (and stdout).
"""

from __future__ import annotations

import copy
import logging
import sys

import torch

from lerobot.scripts import lerobot_train


def _extract_custom_args() -> dict:
    defaults = {
        "val_dataset.repo_id": None,
        "val_dataset.root": None,
        "val_freq": 1000,
        "val_batches": 20,
        "val_rollout_batches": 0,
        "val_rollout_scope": "executed",  # "executed" or "full"
        "val_rollout_plot_samples": 4,
    }
    int_keys = {"val_freq", "val_batches", "val_rollout_batches", "val_rollout_plot_samples"}
    out = dict(defaults)
    remaining: list[str] = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        matched = False
        for key in defaults:
            prefix = f"--{key}="
            if a.startswith(prefix):
                out[key] = a[len(prefix):]
                matched = True
                break
            if a == f"--{key}":
                out[key] = args[i + 1]
                i += 1
                matched = True
                break
        if not matched:
            remaining.append(a)
        i += 1
    for k in int_keys:
        out[k] = int(out[k])
    sys.argv = [sys.argv[0]] + remaining
    return out


VAL_ARGS = _extract_custom_args()
STATE: dict = {"step": 0}


def _patch_make_dataset() -> None:
    orig = lerobot_train.make_dataset

    def patched(cfg):
        ds = orig(cfg)
        if STATE.get("val_dataset") is None and VAL_ARGS["val_dataset.repo_id"]:
            val_cfg = copy.deepcopy(cfg)
            val_cfg.dataset.repo_id = VAL_ARGS["val_dataset.repo_id"]
            if VAL_ARGS["val_dataset.root"]:
                val_cfg.dataset.root = VAL_ARGS["val_dataset.root"]
            val_cfg.dataset.episodes = None
            # Disable training-time augmentations on the validation set.
            if hasattr(val_cfg.dataset, "recolor_prob"):
                val_cfg.dataset.recolor_prob = 0.0
            if hasattr(val_cfg.dataset, "recolor_white_prob"):
                val_cfg.dataset.recolor_white_prob = 0.0
            if hasattr(val_cfg.dataset, "masks_dir"):
                val_cfg.dataset.masks_dir = None
            STATE["val_dataset"] = orig(val_cfg)
            STATE["val_batch_size"] = cfg.batch_size
            STATE["val_num_workers"] = min(cfg.num_workers, 4)
            logging.info(
                "[validation] Loaded val dataset '%s' with %d frames / %d episodes",
                VAL_ARGS["val_dataset.repo_id"],
                STATE["val_dataset"].num_frames,
                STATE["val_dataset"].num_episodes,
            )
        return ds

    lerobot_train.make_dataset = patched


def _patch_make_policy() -> None:
    orig = lerobot_train.make_policy

    def patched(*args, **kwargs):
        policy = orig(*args, **kwargs)
        STATE["policy_ref"] = policy
        return policy

    lerobot_train.make_policy = patched


def _patch_make_pre_post_processors() -> None:
    orig = lerobot_train.make_pre_post_processors

    def patched(*args, **kwargs):
        pre, post = orig(*args, **kwargs)
        STATE["preprocessor"] = pre
        return pre, post

    lerobot_train.make_pre_post_processors = patched


def _build_val_loader() -> None:
    ds = STATE["val_dataset"]
    STATE["val_loader"] = torch.utils.data.DataLoader(
        ds,
        batch_size=STATE["val_batch_size"],
        num_workers=STATE["val_num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=STATE["val_num_workers"] > 0,
    )
    STATE["val_iter"] = iter(STATE["val_loader"])


def _next_val_batch():
    try:
        return next(STATE["val_iter"])
    except StopIteration:
        STATE["val_iter"] = iter(STATE["val_loader"])
        return next(STATE["val_iter"])


def _predict_chunk(policy, batch):
    """Run the model's action-chunk sampler bypassing the inference-time obs queue.

    The val dataset already returns observations stacked over `n_obs_steps`
    via delta_indices, so we can skip `predict_action_chunk`'s queue-stacking
    branch (which crashes when the deployment-time queue is empty).
    """
    unwrapped = policy
    prepared = unwrapped._prepare_batch(dict(batch))
    return unwrapped._generate_actions(prepared)


def _plot_action_chunks(pred, gt, step: int, n_samples: int):
    """Return a list of wandb.Image objects, one per sample, of pred vs GT per joint."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb
    except Exception as e:
        logging.debug("[validation] plotting skipped: %s", e)
        return []
    n = min(n_samples, pred.shape[0])
    horizon = pred.shape[1]
    n_dim = pred.shape[2]
    t = list(range(horizon))
    images = []
    for s in range(n):
        ncols = min(3, n_dim)
        nrows = (n_dim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows), squeeze=False)
        for d in range(n_dim):
            ax = axes[d // ncols][d % ncols]
            ax.plot(t, gt[s, :, d], color="C0", label="gt", linewidth=1.5)
            ax.plot(t, pred[s, :, d], color="C3", linestyle="--", label="pred", linewidth=1.5)
            ax.set_title(f"dim {d}", fontsize=9)
            ax.tick_params(labelsize=8)
            if d == 0:
                ax.legend(fontsize=8, loc="best")
        for d in range(n_dim, nrows * ncols):
            axes[d // ncols][d % ncols].axis("off")
        fig.suptitle(f"step {step} · sample {s}", fontsize=10)
        fig.tight_layout()
        images.append(wandb.Image(fig))
        plt.close(fig)
    return images


def _run_action_rollout(policy, accelerator, step: int) -> dict:
    """Compare sampled action chunks to ground-truth chunks from the val dataset."""
    from lerobot.utils.constants import ACTION

    n_batches = VAL_ARGS["val_rollout_batches"]
    scope = VAL_ARGS["val_rollout_scope"]
    plot_n = VAL_ARGS["val_rollout_plot_samples"]
    pre = STATE["preprocessor"]
    val_ds = STATE["val_dataset"]
    was_training = policy.training
    policy.eval()

    mse_sum = 0.0
    l1_sum = 0.0
    n_seen = 0
    per_dim_mse_sum = None
    plot_pred = None
    plot_gt = None

    with torch.no_grad(), accelerator.autocast():
        for b_idx in range(n_batches):
            batch = _next_val_batch()
            for cam_key in val_ds.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = pre(batch)
            if ACTION not in batch:
                logging.warning("[validation] rollout: batch missing %r key, skipping", ACTION)
                break
            gt = batch[ACTION]  # (B, horizon, action_dim) — normalized
            pred = _predict_chunk(policy, batch)  # (B, horizon, action_dim) — normalized
            if pred.shape != gt.shape:
                # Truncate to common length to be defensive across policies.
                h = min(pred.shape[1], gt.shape[1])
                pred = pred[:, :h]
                gt = gt[:, :h]
            if scope == "executed":
                n_act = getattr(policy.config, "n_action_steps", pred.shape[1])
                pred_s = pred[:, :n_act]
                gt_s = gt[:, :n_act]
            else:
                pred_s = pred
                gt_s = gt
            diff = (pred_s - gt_s).float()
            mse_sum += float(diff.pow(2).mean().item())
            l1_sum += float(diff.abs().mean().item())
            per_dim = diff.pow(2).mean(dim=(0, 1))  # (action_dim,)
            per_dim_mse_sum = per_dim if per_dim_mse_sum is None else per_dim_mse_sum + per_dim
            n_seen += 1
            if b_idx == 0 and plot_n > 0:
                plot_pred = pred_s.detach().float().cpu().numpy()
                plot_gt = gt_s.detach().float().cpu().numpy()

    if was_training:
        policy.train()
    if n_seen == 0:
        return {}

    log_dict = {
        f"val/rollout_{scope}/action_mse": mse_sum / n_seen,
        f"val/rollout_{scope}/action_l1": l1_sum / n_seen,
    }
    if per_dim_mse_sum is not None:
        per_dim_mse = (per_dim_mse_sum / n_seen).detach().float().cpu().tolist()
        for d, v in enumerate(per_dim_mse):
            log_dict[f"val/rollout_{scope}/action_mse_dim{d}"] = float(v)

    if plot_pred is not None and plot_n > 0:
        images = _plot_action_chunks(plot_pred, plot_gt, step, plot_n)
        if images:
            log_dict[f"val/rollout_{scope}/action_chunks"] = images

    logging.info(
        "[val step=%d] rollout %s mse=%.5f l1=%.5f (%d batches)",
        step,
        scope,
        log_dict[f"val/rollout_{scope}/action_mse"],
        log_dict[f"val/rollout_{scope}/action_l1"],
        n_seen,
    )
    return log_dict


def _run_validation(policy, accelerator, step: int) -> None:
    if "val_loader" not in STATE:
        _build_val_loader()
    pre = STATE["preprocessor"]
    val_ds = STATE["val_dataset"]
    was_training = policy.training
    policy.eval()
    loss_sum = 0.0
    comp_sums: dict[str, float] = {}
    n = 0
    with torch.no_grad(), accelerator.autocast():
        for _ in range(VAL_ARGS["val_batches"]):
            batch = _next_val_batch()
            for cam_key in val_ds.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = pre(batch)
            loss, output_dict = policy.forward(batch)
            loss_sum += float(loss.item())
            if output_dict:
                for k, v in output_dict.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
                    elif isinstance(v, (int, float)):
                        comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
            n += 1
    if was_training:
        policy.train()
    if n == 0:
        return
    avg_loss = loss_sum / n
    log_dict = {"val/loss": avg_loss}
    for k, s in comp_sums.items():
        log_dict[f"val/{k}"] = s / n
    logging.info(
        "[val step=%d] loss=%.4f (%d batches)%s",
        step,
        avg_loss,
        n,
        "".join(
            f" {k}={v:.4f}"
            for k, v in log_dict.items()
            if k != "val/loss" and isinstance(v, (int, float))
        ),
    )

    if VAL_ARGS["val_rollout_batches"] > 0:
        try:
            rollout_log = _run_action_rollout(policy, accelerator, step)
            log_dict.update(rollout_log)
        except Exception as e:
            logging.warning("[validation] rollout failed: %s", e, exc_info=True)

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(log_dict, step=step)
    except Exception as e:  # pragma: no cover
        logging.debug("wandb log skipped: %s", e)


def _patch_update_policy() -> None:
    orig = lerobot_train.update_policy

    def patched(train_metrics, policy, batch, *args, accelerator, **kwargs):
        result = orig(train_metrics, policy, batch, *args, accelerator=accelerator, **kwargs)
        STATE["step"] += 1
        step = STATE["step"]
        val_freq = VAL_ARGS["val_freq"]
        if (
            val_freq > 0
            and step % val_freq == 0
            and STATE.get("val_dataset") is not None
            and STATE.get("preprocessor") is not None
            and accelerator.is_main_process
        ):
            _run_validation(policy, accelerator, step)
        return result

    lerobot_train.update_policy = patched


def main() -> None:
    if VAL_ARGS["val_dataset.repo_id"] is None:
        logging.warning(
            "[validation] --val_dataset.repo_id not set; running plain training "
            "with no validation."
        )
    _patch_make_dataset()
    _patch_make_policy()
    _patch_make_pre_post_processors()
    _patch_update_policy()
    lerobot_train.main()


if __name__ == "__main__":
    main()
