"""
Sentinel Optuna ASHA Sweep Orchestrator.
Runs parallel hyperparameter trials with early pruning to free GPU compute immediately.
"""

import argparse
import sys
import time
import optuna
import yaml
from pathlib import Path

def objective(trial, metric_name="val_loss", max_steps=10):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    print(f"\n[Trial #{trial.number}] START: lr={lr:.6f}, batch_size={batch_size}")

    # Simulated step-by-step training curve
    # Good trials (near optimal lr ~ 0.001) descend smoothly.
    # Bad trials (too high/low lr) diverge or stall.
    optimal_lr = 0.001
    lr_penalty = abs(lr - optimal_lr) * 20.0

    current_loss = 2.5 + lr_penalty

    for step in range(1, max_steps + 1):
        # Simulate step compute
        time.sleep(0.02)
        # Update loss
        decay = 0.85 if lr_penalty < 0.5 else 1.05
        current_loss *= decay

        # Report to Optuna for ASHA pruning evaluation
        trial.report(current_loss, step=step)

        # Check early pruning condition
        if trial.should_prune():
            print(f"[Trial #{trial.number}] ✂️ PRUNED early at step {step}/{max_steps} (loss={current_loss:.4f})")
            raise optuna.TrialPruned()

        print(f"[Trial #{trial.number}] Step {step}/{max_steps} -> {metric_name}: {current_loss:.4f}")

    print(f"[Trial #{trial.number}] 🏆 COMPLETED: Final {metric_name} = {current_loss:.4f}")
    return current_loss

def run_sweep(n_trials=10, n_jobs=2):
    # Configure ASHA pruner (Successive Halving)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2,
        min_early_stopping_rate=0,
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="sentinel-sweep-asha",
    )

    print(f"🚀 Starting Sentinel Optuna ASHA Sweep ({n_trials} trials, {n_jobs} parallel workers)...")
    study.optimize(lambda trial: objective(trial), n_trials=n_trials, n_jobs=n_jobs)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n" + "=" * 60)
    print("📊 Sentinel Sweep Summary:")
    print(f"   Total Trials:     {len(study.trials)}")
    print(f"   Completed:        {len(completed_trials)}")
    print(f"   Pruned Early:     {len(pruned_trials)}")
    print(f"   Best Trial:       #{study.best_trial.number}")
    print(f"   Best Value:       {study.best_value:.4f}")
    print(f"   Best Params:      {study.best_params}")
    print("=" * 60)

    assert len(pruned_trials) > 0, "ASHA pruner should prune at least one losing trial!"
    assert len(completed_trials) > 0, "At least one winning trial should complete!"
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel Optuna ASHA Sweep")
    parser.add_argument("--trials", type=int, default=10, help="Total trials")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker slots")
    args = parser.parse_args()

    run_sweep(n_trials=args.trials, n_jobs=args.workers)
