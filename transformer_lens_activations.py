#!/usr/bin/env python3
"""
TransformerLens Activation Analysis - Based on Main Demo

This script demonstrates how to use TransformerLens to examine activations in transformer models,
following the structure and examples from the TransformerLens main demo.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import wandb


# Import visualization libraries
import holoviews as hv

# Import transformer_lens
from tqdm.notebook import tqdm
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from transformer_lens import evals
from transformer_lens.utils import get_act_name

import transformer_lens.loading_from_pretrained

from holoviews import opts

transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)


# Configure Holoviews to use Bokeh backend
hv.extension("bokeh")

# Turn off automatic differentiation to save GPU memory
torch.set_grad_enabled(False)


def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load a pre-trained model using TransformerLens."""
    # Suppress the LayerNorm warning by setting center_writing_weights to False
    model = HookedTransformer.from_pretrained(model_name, device="cuda")
    print(f"Loaded model: {model_name}")
    print(f"Model has {model.cfg.n_layers} layers")
    print(f"Model has {model.cfg.n_heads} attention heads per layer")
    print(f"Model has {model.cfg.d_model} dimensions")
    print(f"Model uses normalization: {model.cfg.normalization_type}")
    return model


# Plotting helper functions using Bokeh/Holoviews
def show_attention(tensor, tokens=None, title="", **kwargs):
    """Display attention patterns with token labels using Holoviews."""
    data = utils.to_numpy(tensor)

    # Create token labels if provided
    if tokens is not None:
        # Convert tokens to strings for labels
        if hasattr(tokens, "tolist"):
            token_strings = [str(t) for t in tokens.tolist()]
        else:
            token_strings = [str(t) for t in tokens]

        # Handle duplicate tokens by adding occurrence indicators
        token_counts = {}
        token_labels = []

        for token in token_strings:
            if token in token_counts:
                token_counts[token] += 1
                token_labels.append(f"{token}[{token_counts[token] - 1}]")
            else:
                token_counts[token] = 1
                token_labels.append(token)
    else:
        # Use position indices as labels
        token_labels = [str(i) for i in range(data.shape[0])]

    # Create coordinate grid for heatmap
    seq_len = data.shape[0]
    key_positions = []
    query_positions = []
    attention_values = []

    for i in range(seq_len):
        for j in range(seq_len):
            key_positions.append(token_labels[j])  # Key position (columns)
            query_positions.append(token_labels[i])  # Query position (rows)
            attention_values.append(data[i, j])

    # Create DataFrame for heatmap
    df = pd.DataFrame(
        {
            "Key Position": key_positions,
            "Query Position": query_positions,
            "Attention": attention_values,
        }
    )

    # Create Holoviews HeatMap
    plot = hv.HeatMap(df, kdims=["Key Position", "Query Position"], vdims=["Attention"])

    # Style the plot
    plot = plot.opts(
        cmap="RdBu_r",
        colorbar=True,
        title=title,
        width=600,
        height=500,
        tools=["hover"],  # Enable hover tooltips
        **kwargs,
    )

    # Display the plot
    hv.render(plot, backend="bokeh")
    return plot


def bar(tensor, title="", xaxis="", yaxis="", **kwargs):
    """Display a tensor as a bar plot using Holoviews."""
    data = utils.to_numpy(tensor)
    df = pd.DataFrame({"values": data, "index": range(len(data))})

    # Create bar plot using hvplot
    plot = df.hvplot.bar(
        x="index", y="values", title=title, xlabel=xaxis, ylabel=yaxis, **kwargs
    )

    # Display the plot
    hv.render(plot, backend="bokeh")
    return plot


def attention_analysis(model, cache, tokens, layer_idx=0, head_idx=0):
    """Analyze attention patterns following the demo."""
    print("=== Attention Analysis ===\n")

    # Extract attention patterns
    attn_patterns = cache[f"blocks.{layer_idx}.attn.hook_attn_scores"]
    print(f"Attention patterns shape: {attn_patterns.shape}")
    print("Shape breakdown: [batch, heads, seq_len, seq_len]")

    # Get attention for specific head
    head_attention = attn_patterns[0, head_idx]  # [seq_len, seq_len]
    print(f"Head {head_idx} attention shape: {head_attention.shape}\n")

    # Visualize attention patterns with token labels
    token_strings = [model.to_string(t) for t in tokens[0]]

    # Using Holoviews for interactive visualization
    show_attention(
        head_attention,
        tokens=token_strings,
        title=f"Layer {layer_idx}, Head {head_idx} Attention",
    )

    return head_attention


def show_all_attention_heads(
    model, cache, text, layer_idx=0, heads=None, cols=4, axiswise=True
):
    """Create attention visualizations for specified heads in a grid layout."""
    if heads is None:
        heads = list(range(model.cfg.n_heads))

    tokens = model.to_str_tokens(text)

    # Generate attention patterns for specified heads
    attention_maps = []
    for head_idx in heads:
        attn = attention_analysis(
            model, cache, model.to_tokens(text), head_idx=head_idx, layer_idx=layer_idx
        )
        attn[attn == -np.inf] = np.nan

        # Use show_attention method to create visualization
        heatmap = show_attention(attn, tokens=tokens).opts(
            title=f"Head {head_idx}", width=500, height=400, xrotation=45
        )
        attention_maps.append(heatmap)

    # Create grid layout
    layout = (
        hv.Layout(attention_maps)
        .cols(cols)
        .opts(title=f"Attention Patterns for Heads {heads} (Layer {layer_idx})")
        .opts(opts.HeatMap(axiswise=axiswise))
    )

    return layout


# Linear Probe Functions for Model Analysis


def create_linear_probe(activation_dim, num_classes, device="cpu"):
    """Create a linear probe (classifier) for analyzing activations."""
    probe = torch.nn.Linear(activation_dim, num_classes, bias=True)
    probe.to(device)
    return probe


def extract_activations_for_probe(
    model,
    texts: list[str],
    labels: list[int],
    activation_name="resid_post",
    pool=False,
    pbar=True,
):
    """Extract activations and labels for training a linear probe."""
    all_activations = []
    all_labels = []

    for text, label in tqdm(
        list(zip(texts, labels)), disable=not pbar, desc="Extracting activations"
    ):
        # Tokenize and get activations
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            cache = cache.to("cpu")

        # Get activations from specified layer
        activations = torch.stack(
            [
                cache[get_act_name(activation_name, layer)]
                for layer in range(model.cfg.n_layers)
            ],
            dim=1,
        )  # [batch, n_layers, seq_len, d_model]

        # Use mean pooling over sequence length or last token
        pooled_activations = (
            activations[:, :, -1] if not pool else activations.mean(dim=2)
        ).reshape(activations.shape[0], -1)  # [batch, n_layers * d_model]

        # Convert to numpy and back to completely detach from computation graph
        pooled_activations = torch.tensor(
            pooled_activations.cpu().numpy(), dtype=torch.float32
        )

        all_activations.append(pooled_activations)
        all_labels.extend([label] * pooled_activations.shape[0])

    with torch.no_grad():
        # Concatenate all activations
        X = torch.cat(all_activations, dim=0)  # [total_samples, n_layers * d_model]
        y = torch.tensor(all_labels, dtype=torch.long)

    # Final check - ensure no gradients
    assert not X.requires_grad, "Activations should not require gradients"
    assert not y.requires_grad, "Labels should not require gradients"

    return X, y


def extract_activations(
    model,
    probe_tasks,
    activation_name: str = "resid_post",
    pool: bool = False,
    pbar: bool = True,
):
    """Extract activations for a probe task."""
    return {
        task_name: extract_activations_for_probe(
            model,
            probe_tasks[task_name]["texts"],
            probe_tasks[task_name]["labels"],
            activation_name,
            pool,
            pbar,
        )
        for task_name in probe_tasks.keys()
    }


def train_linear_probe_for_task(
    model,
    probe_tasks,
    task_name: str,
    data: dict[str, tuple[torch.Tensor, torch.Tensor]],
    learning_rate=0.01,
    epochs=100,
    device="cpu",
    pbar=True,
    weight_decay=0,
    l1_lambda=0.0,
    batch_size=32,
    apply_pca=True,
    pca_components=100,
    val_report_frequency=20,
    use_wandb=True,
    wandb_project="interpretability-probes",
    wandb_run_name=None,
    init_normal=True,
    train_test_split: float = 0.8,
):
    """Train a linear probe on activations using stochastic batch gradient descent.

    Args:
        model: The transformer model
        probe_tasks: Dictionary of probe tasks
        task_name: Name of the task to train on
        data: Dictionary containing training data
        learning_rate: Learning rate for optimizer (default: 0.01)
        epochs: Number of training epochs (default: 100)
        device: Device to train on (default: "cpu")
        pbar: Whether to show progress bar (default: True)
        weight_decay: L2 regularization strength (default: 0)
        l1_lambda: L1 regularization strength (default: 0.0, no L1 regularization)
        batch_size: Batch size for stochastic gradient descent (default: 32)
        apply_pca: Whether to apply PCA for dimensionality reduction (default: True)
        pca_components: Number of PCA components to retain (default: 100)
        val_report_frequency: How often to print validation metrics (default: 20)
        use_wandb: Whether to log metrics to wandb (default: True)
        wandb_project: wandb project name (default: "interpretability-probes")
        wandb_run_name: wandb run name (default: None, auto-generated)
        init_normal: Whether to initialize weights to 1 and bias to 0 (default: True)
    """

    X, y = data[task_name]
    print(f"Original data shape: {X.shape}")
    print(f"Data labels shape: {y.shape}")

    # Apply PCA for dimensionality reduction if requested
    if apply_pca:
        print(f"Applying PCA to reduce dimensions to {pca_components}...")
        pca = PCA(n_components=pca_components)
        X_numpy = X.cpu().numpy()
        X_pca = pca.fit_transform(X_numpy)
        X = torch.tensor(X_pca, dtype=torch.float32)
        print(f"PCA data shape: {X.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

        # Update input dimension for probe
        input_dim = pca_components
    else:
        input_dim = model.cfg.d_model * model.cfg.n_layers

    # Initialize wandb if requested
    if use_wandb:
        run_name = wandb_run_name or f"{task_name}_probe"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "task_name": task_name,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "l1_lambda": l1_lambda,
                "apply_pca": apply_pca,
                "pca_components": pca_components if apply_pca else None,
                "input_dim": input_dim,
                "num_classes": len(set(probe_tasks[task_name]["labels"])),
                "model_name": getattr(model.cfg, "model_name", "unknown"),
                "n_layers": model.cfg.n_layers,
                "d_model": model.cfg.d_model,
                "device": device,
                "init_normal": init_normal,
            },
        )

    # Temporarily enable gradients for probe training
    torch.set_grad_enabled(True)

    # Create probe with gradients enabled
    probe = torch.nn.Linear(
        input_dim,
        len(set(probe_tasks[task_name]["labels"])),
        bias=False,
        device=device,
    )

    # Initialize weights and bias if requested
    if init_normal:
        with torch.no_grad():
            torch.nn.init.normal_(probe.weight, mean=0.0, std=1.0 / np.sqrt(input_dim))
            if probe.bias is not None:
                torch.nn.init.zeros_(probe.bias)

    # Ensure probe parameters require gradients
    for param in probe.parameters():
        param.requires_grad = True

    # Split data properly
    n_samples = len(X)
    split_idx = int(train_test_split * n_samples)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:].to(device)
    y_val = y[split_idx:].to(device)

    # Create DataLoader for training batches
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in tqdm(
        range(epochs), desc=f"Training {task_name} probe", disable=not pbar
    ):
        # Training with batches
        probe.train()
        epoch_train_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = probe(batch_X)
            loss = criterion(outputs, batch_y)

            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_penalty = sum(param.abs().sum() for param in probe.parameters())
                loss = loss + l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                train_correct_predictions += (predicted == batch_y).sum().item()
                train_total_samples += batch_y.size(0)

        avg_train_loss = epoch_train_loss / num_batches
        train_accuracy = train_correct_predictions / train_total_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        probe.eval()
        with torch.no_grad():
            val_outputs = probe(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            # Calculate accuracy
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item()
            val_accuracies.append(accuracy)

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss/train": avg_train_loss,
                    "loss/val": val_loss.item(),
                    "accuracy/train": train_accuracy,
                    "accuracy/val": accuracy,
                }
            )

        if epoch % val_report_frequency == 0:
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                f"Val Loss = {val_loss.item():.4f}, Val Acc = {accuracy:.4f}"
            )

    torch.set_grad_enabled(False)

    # Log final metrics to wandb
    if use_wandb:
        wandb.log(
            {
                "final/train_loss": train_losses[-1],
                "final/train_accuracy": train_accuracies[-1],
                "final/val_loss": val_losses[-1],
                "final/val_accuracy": val_accuracies[-1],
                "best/train_accuracy": max(train_accuracies),
                "best/val_accuracy": max(val_accuracies),
            }
        )
        # Log PCA explained variance if applicable
        if apply_pca:
            wandb.log({"pca_explained_variance": pca.explained_variance_ratio_.sum()})

        wandb.finish()

    return probe, {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_accuracy": val_accuracies[-1],
    }


def analyze_probe_performance(probe, X_test, y_test, class_names=None, device="cpu"):
    """Analyze the performance of a trained linear probe."""
    probe.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        outputs = probe(X_test)
        probabilities = torch.softmax(outputs, dim=-1)
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        accuracy = (predicted == y_test).float().mean().item()

        # Calculate per-class accuracy
        unique_classes = torch.unique(y_test)
        per_class_accuracy = {}

        for cls in unique_classes:
            cls_mask = y_test == cls
            if cls_mask.sum() > 0:
                cls_acc = (
                    (predicted[cls_mask] == y_test[cls_mask]).float().mean().item()
                )
                class_name = (
                    class_names[cls.item()] if class_names else f"Class {cls.item()}"
                )
                per_class_accuracy[class_name] = cls_acc

        # Get confidence scores
        confidence = probabilities.max(dim=1)[0].mean().item()

        print("=== Probe Performance Analysis ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Average Confidence: {confidence:.4f}")
        print("\nPer-class Accuracy:")
        for class_name, acc in per_class_accuracy.items():
            print(f"  {class_name}: {acc:.4f}")

        return {
            "accuracy": accuracy,
            "confidence": confidence,
            "per_class_accuracy": per_class_accuracy,
            "predictions": predicted.cpu(),
            "probabilities": probabilities.cpu(),
        }


def visualize_probe_results(probe_results):
    """Visualize the results of all probes."""
    print("=== Probe Results Visualization ===\n")

    for task_name, layer_results in probe_results.items():
        print(f"Task: {task_name}")

        # Extract accuracies for each layer
        layers = []
        accuracies = []

        for layer_idx, results in layer_results.items():
            layers.append(layer_idx)
            accuracies.append(results["performance"]["accuracy"])

        # Create bar plot
        bar(
            torch.tensor(accuracies),
            title=f"Probe Accuracy by Layer - {task_name}",
            xaxis="Layer",
            yaxis="Accuracy",
        )

        print(f"Layer accuracies: {[f'{acc:.3f}' for acc in accuracies]}")
        print()


def create_probe_activation_hook(probe_path, factor=1.0, device="cuda", opposite=False):
    """
    Create a TransformerLens activation hook that adds probe activations to resid_post.

    Args:
        probe_path: Path to the probe weights file
        factor: Scaling factor for the probe activations
        device: Device to load probe on

    Returns:
        Hook function that can be used with model.run_with_hooks()
    """
    # Load the probe weights
    probe = torch.load(probe_path, map_location=device)

    # Extract weight matrix from the probe
    if isinstance(probe, dict):
        weights = probe.get("weight", probe)
    else:
        weights = probe.weight if hasattr(probe, "weight") else probe

    # If 2D, take the second element (assuming first dim is classes)
    if weights.dim() == 2:
        weights = weights[0 if opposite else 1]  # Take second class weights

    def probe_hook(activation, hook):
        """
        Hook function to add probe activations to residual stream.

        Args:
            activation: The residual stream activation [batch, seq_len, d_model]
            hook: The hook object containing layer information
        """
        batch_size, seq_len, d_model = activation.shape
        layer_idx = hook.layer()
        n_layers = weights.shape[0] // d_model

        # Extract the relevant portion of probe weights for this layer
        start_idx = layer_idx * d_model
        end_idx = (layer_idx + 1) * d_model
        layer_weights = weights[start_idx:end_idx]  # [d_model]

        # Expand for batch and sequence dimensions
        probe_addition = (
            layer_weights.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        )

        # Scale and add to the original activation
        return activation + (probe_addition * factor)

    return probe_hook


def get_example_probe_tasks(sample_size=500):
    """Get example probe tasks using MATH dataset categories."""
    from datasets import load_dataset

    # Load MATH dataset categories
    ds_algebra = load_dataset("EleutherAI/hendrycks_math", "algebra")
    ds_geometry = load_dataset("EleutherAI/hendrycks_math", "geometry")
    ds_counting_and_probability = load_dataset(
        "EleutherAI/hendrycks_math", "counting_and_probability"
    )
    ds_intermediate_algebra = load_dataset(
        "EleutherAI/hendrycks_math", "intermediate_algebra"
    )
    ds_number_theory = load_dataset("EleutherAI/hendrycks_math", "number_theory")
    ds_prealgebra = load_dataset("EleutherAI/hendrycks_math", "prealgebra")
    ds_precalculus = load_dataset("EleutherAI/hendrycks_math", "precalculus")

    # Combine all problems and levels
    all_problems = (
        ds_prealgebra["train"]["problem"][:]
        + ds_algebra["train"]["problem"][:]
        + ds_precalculus["train"]["problem"][:]
        + ds_counting_and_probability["train"]["problem"][:]
        + ds_intermediate_algebra["train"]["problem"][:]
        + ds_number_theory["train"]["problem"][:]
        + ds_geometry["train"]["problem"][:]
    )

    all_levels = (
        ds_prealgebra["train"]["level"][:]
        + ds_algebra["train"]["level"][:]
        + ds_precalculus["train"]["level"][:]
        + ds_counting_and_probability["train"]["level"][:]
        + ds_intermediate_algebra["train"]["level"][:]
        + ds_number_theory["train"]["level"][:]
        + ds_geometry["train"]["level"][:]
    )
    # Get indices for level 0 and 4 using list comprehensions
    easy_idxs = [
        i
        for i, level in enumerate(all_levels)
        if "?" not in level and int(level.strip("Level ")) - 1 == 0
    ]
    hard_idxs = [
        i
        for i, level in enumerate(all_levels)
        if "?" not in level and int(level.strip("Level ")) - 1 == 4
    ]

    # Combine and limit to sample_size
    selected_idxs = easy_idxs[: sample_size // 2] + hard_idxs[: sample_size // 2]

    selected_problems = [all_problems[i] for i in selected_idxs]
    selected_idxs_rand = np.random.permutation(len(selected_idxs))

    return {
        "geometry": {
            "texts": (
                ds_algebra["train"]["problem"][:][: sample_size // 10]
                + ds_geometry["train"]["problem"][:][: sample_size // 2]
                + ds_counting_and_probability["train"]["problem"][:][
                    : sample_size // 10
                ]
                + ds_intermediate_algebra["train"]["problem"][:][: sample_size // 10]
                + ds_number_theory["train"]["problem"][: sample_size // 5]
                # + ds_prealgebra["train"]["problem"][: sample_size // 10]
                # + ds_precalculus["train"]["problem"][:sample_size]
            ),
            "labels": (
                [0] * (sample_size // 10)  # Algebra
                + [1] * (sample_size // 2)  # Geometry
                + [0] * (sample_size // 10)  # Counting & Probability
                + [0] * (sample_size // 10)  # Intermediate Algebra
                + [0] * (sample_size // 5)  # Number Theory
                # + [0] * (sample_size // 10)  # Prealgebra
                # + [6] * sample_size  # Precalculus
            ),
            "class_names": [
                "Algebra",
                "Geometry",
                "Counting & Probability",
                "Intermediate Algebra",
                "Number Theory",
                "Prealgebra",
                "Precalculus",
            ],
        },
        "difficulty": {
            "texts": [selected_problems[j] for j in selected_idxs_rand],
            "labels": np.array([0] * (sample_size // 2) + [1] * (sample_size // 2))[
                selected_idxs_rand
            ],  # 0 for easy, 1 for hard
            "class_names": ["Easy", "Hard"],
        },
    }
