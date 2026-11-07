"""Sentinel Native Explainable AI (xAI) Engine.

Zero-dependency, high-performance model interpretability using native PyTorch autograd:
- Integrated Gradients (Axiomatic Attribution for tabular, sequence, and embeddings)
- Grad-CAM (Visual Class Activation Mapping for convolutional layers)
- Spectral Rank & Representation Collapse Diagnostics (SVD Entropy)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def integrated_gradients(
    model,
    input_tensor,
    target_class: Optional[int] = None,
    baseline=None,
    steps: int = 25,
) -> Tuple[Any, float]:
    r"""Compute feature attributions using Integrated Gradients.

    Attr_i = (x_i - x_0,i) * (1/m) sum_{k=1}^m \nabla_{x_i} f(x_0 + (k/m)*(x - x_0))
    Satisfies completeness: sum(attributions) ~= f(x) - f(baseline).
    """
    import torch

    model.eval()
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    input_tensor = input_tensor.to(device)

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    elif not isinstance(baseline, torch.Tensor):
        baseline = torch.tensor(baseline, dtype=torch.float32, device=device)
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0)
    else:
        baseline = baseline.to(device)

    # Determine target class from baseline or input prediction if not specified
    with torch.no_grad():
        orig_out = model(input_tensor)
        if target_class is None:
            if orig_out.dim() > 1 and orig_out.shape[-1] > 1:
                target_class = int(orig_out.argmax(dim=-1).item())
            else:
                target_class = 0
        base_out = model(baseline)
        target_score = float(orig_out[0, target_class].item() if orig_out.dim() > 1 else orig_out[0].item())
        base_score = float(base_out[0, target_class].item() if base_out.dim() > 1 else base_out[0].item())
        delta_score = target_score - base_score

    # Linear interpolation path between baseline and input
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    interpolated = torch.cat([baseline + alpha * (input_tensor - baseline) for alpha in alphas], dim=0)
    interpolated.requires_grad_(True)

    outputs = model(interpolated)
    if outputs.dim() > 1 and outputs.shape[-1] > 1:
        selected = outputs[:, target_class].sum()
    else:
        selected = outputs.sum()

    grads = torch.autograd.grad(selected, interpolated)[0]
    avg_grads = grads.mean(dim=0, keepdim=True)
    attributions = (input_tensor - baseline) * avg_grads

    return attributions.detach().cpu(), delta_score


def spectral_rank(weight: Any) -> Dict[str, float]:
    r"""Compute effective spectral rank via Singular Value Decomposition (SVD).

    Effective rank measures the dimensionality of learned representations:
    p_i = \sigma_i / sum(\sigma_j)
    H = - sum(p_i * ln(p_i))
    Effective Rank = exp(H)

    If Effective Rank << min(M, N), the layer has suffered representation collapse.
    """
    import torch

    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight, dtype=torch.float32)

    # If conv2d weights (C_out, C_in, kH, kW), flatten to 2D
    if weight.dim() > 2:
        weight = weight.flatten(start_dim=1)
    elif weight.dim() == 1:
        weight = weight.unsqueeze(0)

    try:
        _, s, _ = torch.linalg.svd(weight.float(), full_matrices=False)
        total_s = s.sum().item()
        if total_s <= 1e-12:
            return {"effective_rank": 1.0, "max_rank": float(min(weight.shape)), "collapse_ratio": 0.0}

        p = s / total_s
        p = p[p > 1e-12]
        entropy = -float((p * torch.log(p)).sum().item())
        eff_rank = math.exp(entropy)
        max_rank = float(min(weight.shape))
        return {
            "effective_rank": round(eff_rank, 2),
            "max_rank": max_rank,
            "collapse_ratio": round(eff_rank / max(max_rank, 1.0), 3),
        }
    except Exception:
        return {"effective_rank": 1.0, "max_rank": float(min(weight.shape)), "collapse_ratio": 1.0}


def explain(
    model,
    input_sample: Any,
    feature_names: Optional[List[str]] = None,
    target_class: Optional[int] = None,
    baseline: Optional[Any] = None,
    steps: int = 25,
) -> Dict[str, Any]:
    """Generate a full explainability report for a sample prediction."""
    import torch

    attr_tensor, delta_score = integrated_gradients(
        model, input_sample, target_class=target_class, baseline=baseline, steps=steps
    )
    raw_attrs = attr_tensor.flatten().tolist()
    sample_vals = input_sample.flatten().tolist() if isinstance(input_sample, torch.Tensor) else list(input_sample)

    num_features = len(raw_attrs)
    if feature_names is None or len(feature_names) != num_features:
        feature_names = [f"feature_{i+1}" for i in range(num_features)]

    abs_sum = sum(abs(a) for a in raw_attrs) or 1e-9
    attributions = []
    for name, val, a in zip(feature_names, sample_vals, raw_attrs):
        attributions.append({
            "feature": name,
            "value": round(float(val), 4),
            "attribution": round(float(a), 5),
            "importance_pct": round(abs(float(a)) / abs_sum * 100.0, 2),
            "direction": "positive" if a > 0 else "negative" if a < 0 else "neutral",
        })

    # Sort by absolute importance
    attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)

    # Collect layer spectral ranks across model
    layer_ranks = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            layer_ranks[name] = spectral_rank(param.detach().cpu())

    return {
        "target_class": target_class,
        "delta_score": round(delta_score, 4),
        "attributions": attributions,
        "top_positive": [x for x in attributions if x["attribution"] > 0][:5],
        "top_negative": [x for x in attributions if x["attribution"] < 0][:5],
        "spectral_ranks": layer_ranks,
    }


def log_explanation(
    explanation: Dict[str, Any],
    out_dir: str = ".sentinel",
    filename: str = "xai_explanation.json",
):
    """Write explanation report to disk for daemon API and dashboard consumption."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    out_file = p / filename
    try:
        out_file.write_text(json.dumps(explanation, indent=2))
    except Exception:
        pass


def grad_cam(
    model,
    input_tensor: Any,
    target_layer_name: Optional[str] = None,
    target_class: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Compute Grad-CAM (Class Activation Map) for convolutional vision models.

    L_{Grad-CAM} = ReLU(sum_k \alpha_k A^k), where \alpha_k = (1/Z) sum_{i,j} \partial y_c / \partial A^k_{i,j}
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    model.eval()
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    input_tensor = input_tensor.to(device)

    # Locate target conv layer if not specified (pick last conv2d in model)
    target_layer = None
    resolved_layer_name = target_layer_name or ""
    if target_layer_name:
        for name, mod in model.named_modules():
            if name == target_layer_name:
                target_layer = mod
                break
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d)):
                target_layer = mod
                resolved_layer_name = name

    if target_layer is None:
        # Fallback to first module with 4D activations
        for name, mod in model.named_modules():
            if list(mod.parameters()) and len(list(mod.children())) == 0:
                target_layer = mod
                resolved_layer_name = name

    if target_layer is None:
        raise ValueError("No suitable convolutional layer found for Grad-CAM.")

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(mod, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        activations.append(t)

    def bwd_hook(mod, grad_in, grad_out):
        g = grad_out[0] if isinstance(grad_out, (list, tuple)) else grad_out
        gradients.append(g)

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        out = model(input_tensor)
        if out.dim() > 1 and out.shape[-1] > 1:
            if target_class is None:
                target_class = int(out.argmax(dim=-1).item())
            loss = out[0, target_class]
            prob = float(torch.softmax(out, dim=-1)[0, target_class].item())
        else:
            if target_class is None:
                target_class = 0
            loss = out.sum()
            prob = float(torch.sigmoid(out)[0].item()) if out.numel() == 1 else 1.0

        loss.backward()

        act = activations[0].detach()
        grad = gradients[0].detach()

        # Global average pool gradients along spatial dimensions (H, W)
        weights = grad.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to match input height and width
        H, W = input_tensor.shape[-2], input_tensor.shape[-1]
        if cam.shape[-2:] != (H, W):
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        heatmap = [[round(float(v), 4) for v in row] for row in cam.tolist()]

        return {
            "type": "grad_cam",
            "layer": resolved_layer_name,
            "target_class": target_class,
            "confidence": round(prob, 4),
            "grid_h": H,
            "grid_w": W,
            "heatmap": heatmap,
        }
    finally:
        h1.remove()
        h2.remove()


def attention_rollout(model, input_tensor: Any) -> Dict[str, Any]:
    r"""Compute Attention Rollout across transformer multi-head self-attention layers.

    R_{l} = \tilde{A}_l \cdot R_{l-1}, where \tilde{A}_l = 0.5 * A_l + 0.5 * I
    """
    import torch
    import torch.nn as nn

    model.eval()
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    input_tensor = input_tensor.to(device)

    attention_maps: List[torch.Tensor] = []

    def make_attn_hook():
        def hook(mod, inp, out):
            # If attention module returns (output, weights)
            if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
                attention_maps.append(out[1].detach().cpu())
            elif isinstance(mod, nn.MultiheadAttention):
                pass
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower() or isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(make_attn_hook()))

    try:
        with torch.no_grad():
            _ = model(input_tensor)

        if not attention_maps:
            # Synthetic rollout from attention weights if model exposes layers
            return {"type": "attention_rollout", "layers_tracked": 0, "rollout": []}

        # Average attention across heads and compute rollout
        rollout = torch.eye(attention_maps[0].shape[-1])
        for attn in attention_maps:
            if attn.dim() == 4:
                attn = attn.mean(dim=1)  # mean over heads
            attn = attn[0]  # first sample in batch
            attn_with_identity = 0.5 * attn + 0.5 * torch.eye(attn.shape[-1])
            attn_norm = attn_with_identity / attn_with_identity.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(attn_norm, rollout)

        cls_rollout = rollout[0].tolist()  # CLS token attention to other tokens
        return {
            "type": "attention_rollout",
            "layers_tracked": len(attention_maps),
            "tokens": len(cls_rollout),
            "rollout": [round(float(v), 5) for v in cls_rollout],
        }
    finally:
        for h in hooks:
            h.remove()


def loss_landscape(
    model,
    inputs: Any,
    targets: Any,
    loss_fn=None,
    grid_size: int = 15,
    range_span: float = 1.0,
) -> Dict[str, Any]:
    r"""Compute filter-normalized 2D loss surface contour (Li et al., 2018).

    L(\alpha, \beta) = \mathcal{L}(W^* + \alpha \delta + \beta \eta)
    with filter-normalization: \delta_{i,j} = d_{i,j} * ||W_{i,j}|| / ||d_{i,j}||
    """
    import torch
    import torch.nn as nn

    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss() if targets.dtype == torch.long or (isinstance(targets, torch.Tensor) and targets.dim() == 1) else nn.MSELoss()

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    else:
        inputs = inputs.to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, device=device)
    else:
        targets = targets.to(device)

    # Save original base parameters W*
    orig_weights = [p.data.clone() for p in model.parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]

    # Generate filter-normalized random directions delta and eta
    delta: List[torch.Tensor] = []
    eta: List[torch.Tensor] = []
    for w in orig_weights:
        d = torch.randn_like(w)
        e = torch.randn_like(w)
        # Filter-wise normalization
        if w.dim() > 1:
            w_norm = w.norm(dim=list(range(1, w.dim())), keepdim=True)
            d_norm = d.norm(dim=list(range(1, d.dim())), keepdim=True) + 1e-7
            e_norm = e.norm(dim=list(range(1, e.dim())), keepdim=True) + 1e-7
            delta.append(d * (w_norm / d_norm))
            eta.append(e * (w_norm / e_norm))
        else:
            w_norm = w.norm()
            delta.append(d * (w_norm / (d.norm() + 1e-7)))
            eta.append(e * (w_norm / (e.norm() + 1e-7)))

    alphas = torch.linspace(-range_span, range_span, grid_size).tolist()
    betas = torch.linspace(-range_span, range_span, grid_size).tolist()
    surface: List[List[float]] = []

    try:
        with torch.no_grad():
            for b in betas:
                row: List[float] = []
                for a in alphas:
                    for p, w, d, e in zip(params, orig_weights, delta, eta):
                        p.data.copy_(w + a * d + b * e)
                    preds = model(inputs)
                    loss_val = float(loss_fn(preds, targets).item())
                    row.append(round(loss_val, 4))
                surface.append(row)
    finally:
        # Restore original model weights
        for p, w in zip(params, orig_weights):
            p.data.copy_(w)

    flat = [v for row in surface for v in row]
    center_idx = grid_size // 2
    center_loss = surface[center_idx][center_idx]
    local_pts = [
        surface[center_idx][center_idx],
        surface[center_idx - 1][center_idx],
        surface[center_idx + 1][center_idx],
        surface[center_idx][center_idx - 1],
        surface[center_idx][center_idx + 1],
    ]
    mean_local = sum(local_pts) / len(local_pts)
    var_local = sum((x - mean_local) ** 2 for x in local_pts) / len(local_pts)
    flatness_score = round(1.0 / (var_local + 1e-4), 2)

    return {
        "grid_size": grid_size,
        "alphas": [round(a, 3) for a in alphas],
        "betas": [round(b, 3) for b in betas],
        "surface": surface,
        "center_loss": center_loss,
        "min_loss": min(flat),
        "max_loss": max(flat),
        "flatness_score": flatness_score,
        "flatness_grade": "Wide Basin (Flat Minima)" if flatness_score > 500.0 else "Convex Slope" if flatness_score > 100.0 else "Sharp Ravine (Sensitive)",
    }

