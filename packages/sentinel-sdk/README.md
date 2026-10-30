# Sentinel Python SDK

Thin Python client for Sentinel ML training supervisor.

```python
import sentinel

# 1. Initialize run
sentinel.init(project="vision-cifar", run_name="resnet18-baseline", config={"lr": 3e-4})

for epoch in range(10):
    loss = train_epoch()
    # 2. Log metrics safely (never crashes training)
    sentinel.log({"train/loss": loss, "epoch": epoch}, step=epoch)

# 3. Finish run
sentinel.finish()
```
