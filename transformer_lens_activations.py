#!/usr/bin/env python3
"""
TransformerLens Activation Analysis - Based on Main Demo

This script demonstrates how to use TransformerLens to examine activations in transformer models,
following the structure and examples from the TransformerLens main demo.
"""

import torch
import numpy as np
import pandas as pd


# Import visualization libraries
import holoviews as hv

# Import transformer_lens
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from transformer_lens import evals
from transformer_lens.utils import get_act_name

import transformer_lens.loading_from_pretrained

transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

from holoviews import opts


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
def imshow(tensor, title="", xaxis="", yaxis="", **kwargs):
    """Display a tensor as an image using Holoviews."""
    data = utils.to_numpy(tensor)

    # Create coordinate arrays
    y_size, x_size = data.shape
    x_coords = np.arange(x_size)
    y_coords = np.arange(y_size)

    # Create Holoviews Image object
    plot = hv.Image(
        (x_coords, y_coords, data),
        kdims=[xaxis or "X", yaxis or "Y"],
        vdims=["Value"],
        label=title,
    )

    # Style the plot
    plot = plot.opts(cmap="RdBu_r", colorbar=True, title=title, **kwargs)

    # Display the plot
    hv.render(plot, backend="bokeh")
    return plot


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


def line(tensor, title="", xaxis="", yaxis="", **kwargs):
    """Display a tensor as a line plot using Holoviews."""
    data = utils.to_numpy(tensor)
    df = pd.DataFrame(data)

    # Create line plot using hvplot
    plot = df.hvplot.line(title=title, xlabel=xaxis, ylabel=yaxis, **kwargs)

    # Display the plot
    hv.render(plot, backend="bokeh")
    return plot


def scatter(x, y, title="", xaxis="", yaxis="", **kwargs):
    """Display a scatter plot using Holoviews."""
    x_data = utils.to_numpy(x)
    y_data = utils.to_numpy(y)
    df = pd.DataFrame({"x": x_data, "y": y_data})

    # Create scatter plot using hvplot
    plot = df.hvplot.scatter(
        x="x", y="y", title=title, xlabel=xaxis, ylabel=yaxis, **kwargs
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


def histogram(tensor, title="", xaxis="", yaxis="", **kwargs):
    """Display a tensor as a histogram using Holoviews."""
    data = utils.to_numpy(tensor)
    df = pd.DataFrame({"values": data})

    # Create histogram using hvplot
    plot = df.hvplot.hist(y="values", title=title, xlabel=xaxis, ylabel=yaxis, **kwargs)

    # Display the plot
    hv.render(plot, backend="bokeh")
    return plot


def basic_activation_analysis(model, input_text="The cat sat on the mat."):
    """Basic activation analysis following the demo structure."""
    print("=== Basic Activation Analysis ===\n")

    # Tokenize input
    tokens = model.to_tokens(input_text)
    print(f"Input text: {input_text}")
    print(f"Tokens: {tokens}")
    print(f"Tokenized text: {model.to_string(tokens[0])}")
    print(f"Number of tokens: {tokens.shape[1]}\n")

    # Get activations with cache
    logits, cache = model.run_with_cache(tokens)
    print(f"Final logits shape: {logits.shape}")
    print(f"Cache keys (first 10): {list(cache.keys())[:10]}...\n")

    return logits, cache, tokens


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


def residual_stream_analysis(model, cache, tokens):
    """Analyze residual stream following the demo."""
    print("=== Residual Stream Analysis ===\n")

    # Get residual stream before and after each layer
    for layer in range(min(3, model.cfg.n_layers)):
        resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]
        resid_post = cache[f"blocks.{layer}.hook_resid_post"]

        print(f"Layer {layer}:")
        print(f"  Residual pre shape: {resid_pre.shape}")
        print(f"  Residual post shape: {resid_post.shape}")
        print(f"  Residual stream change: {torch.norm(resid_post - resid_pre):.4f}")

        # Visualize residual stream changes
        resid_change = resid_post - resid_pre
        imshow(
            resid_change[0],  # First batch
            xaxis="Position",
            yaxis="Feature",
            title=f"Layer {layer} Residual Stream Change",
        )

    print()


def mlp_analysis(model, cache, tokens, layer_idx=0):
    """Analyze MLP activations following the demo."""
    print("=== MLP Analysis ===\n")

    # Get MLP activations
    mlp_activations = cache[f"blocks.{layer_idx}.mlp.hook_post"]
    print(f"MLP activations shape: {mlp_activations.shape}")
    print("Shape breakdown: [batch, seq_len, d_mlp]")

    # Analyze activation patterns across sequence
    mlp_activations_seq = mlp_activations[0]  # [seq_len, d_mlp]
    print(f"MLP activations per token shape: {mlp_activations_seq.shape}")

    # Find most active neurons
    neuron_activations = torch.mean(
        mlp_activations_seq, dim=0
    )  # Average across sequence
    top_neurons = torch.topk(neuron_activations, k=10)
    print(f"\nTop 10 most active neurons in layer {layer_idx}:")
    for i, (neuron_idx, activation) in enumerate(
        zip(top_neurons.indices, top_neurons.values)
    ):
        print(f"  Neuron {neuron_idx}: {activation:.4f}")

    # Visualize neuron activations
    bar(
        neuron_activations[:100],  # First 100 neurons
        title=f"Layer {layer_idx} Neuron Activations (First 100)",
        xaxis="Neuron Index",
        yaxis="Activation",
    )

    return mlp_activations


def cross_input_comparison(model, input_texts):
    """Compare activations across different inputs following the demo."""
    print("=== Cross-Input Comparison ===\n")

    activations_dict = {}

    for text in input_texts:
        tokens = model.to_tokens(text)
        logits, cache = model.run_with_cache(tokens)

        # Store final layer activations
        final_activations = cache["ln_final.hook_normalized"]
        activations_dict[text] = final_activations[0, -1]  # Last token, last position

    print("Final layer activations for last token:")
    for text, activations in activations_dict.items():
        print(f"  '{text}': norm = {torch.norm(activations):.4f}")

    # Compare similarity between activations
    texts = list(activations_dict.keys())
    similarities = []
    text_pairs = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = torch.cosine_similarity(
                activations_dict[texts[i]], activations_dict[texts[j]], dim=0
            )
            similarities.append(similarity.item())
            text_pairs.append(f"{texts[i][:20]}... vs {texts[j][:20]}...")
            print(f"Similarity between '{texts[i]}' and '{texts[j]}': {similarity:.4f}")

    # Visualize similarities
    bar(
        torch.tensor(similarities),
        title="Activation Similarities Between Inputs",
        xaxis="Text Pair",
        yaxis="Cosine Similarity",
    )

    return activations_dict


def custom_hook_example(model, tokens):
    """Demonstrate custom hooks following the demo."""
    print("=== Custom Hook Example ===\n")

    def set_to_zero_hook(tensor, hook):
        """Hook function that sets activations to zero."""
        print(f"Hook {hook.name}: setting to zero")
        return torch.zeros_like(tensor)

    def print_shape_hook(tensor, hook):
        """Hook function that prints activation shapes."""
        print(f"Hook {hook.name}: activation shape {tensor.shape}")
        return tensor

    # Run model with custom hooks
    hook_names = [
        "blocks.0.attn.hook_result",
        "blocks.0.mlp.hook_post",
    ]

    print("Running with custom hooks:")
    logits = model.run_with_hooks(
        tokens, fwd_hooks=[(name, print_shape_hook) for name in hook_names]
    )

    print("\nRunning with zero intervention:")
    logits_zero = model.run_with_hooks(
        tokens, fwd_hooks=[("blocks.0.attn.hook_result", set_to_zero_hook)]
    )

    print(f"Original logits shape: {logits.shape}")
    print(f"Modified logits shape: {logits_zero.shape}")
    print("Custom hooks executed successfully!\n")

    return logits, logits_zero


def induction_head_test(model):
    """Test for induction heads following the demo."""
    print("=== Induction Head Test ===\n")

    try:
        # Test induction loss
        induction_loss = evals.induction_loss(model)
        print(f"Induction loss: {induction_loss:.4f}")

        # Context: Random performance is around ln(20000) ≈ 10
        # Naive strategy gets around ln(384) ≈ 5.95
        if induction_loss < 6.0:
            print("Model shows evidence of induction heads!")
        else:
            print("Model does not show strong evidence of induction heads.")

    except Exception as e:
        print(f"Could not run induction test: {e}")

    print()


def attention_head_analysis(model, cache, tokens, layer_idx=0):
    """Analyze all attention heads in a layer following the demo."""
    print("=== Attention Head Analysis ===\n")

    # Get attention patterns for all heads
    attn_patterns = cache[f"blocks.{layer_idx}.attn.hook_attn_scores"]
    print(f"Attention patterns shape: {attn_patterns.shape}")

    # Analyze each head
    n_heads = attn_patterns.shape[1]
    print(f"Analyzing {n_heads} attention heads in layer {layer_idx}")

    # Calculate attention entropy for each head
    attention_entropies = []
    for head_idx in range(n_heads):
        head_attention = attn_patterns[0, head_idx]  # [seq_len, seq_len]

        # Calculate entropy of attention distribution
        attention_probs = torch.softmax(head_attention, dim=-1)
        entropy = -torch.sum(
            attention_probs * torch.log(attention_probs + 1e-8), dim=-1
        )
        avg_entropy = torch.mean(entropy).item()
        attention_entropies.append(avg_entropy)

        print(f"  Head {head_idx}: avg entropy = {avg_entropy:.4f}")

    # Visualize attention entropies
    bar(
        torch.tensor(attention_entropies),
        title=f"Layer {layer_idx} Attention Head Entropies",
        xaxis="Head Index",
        yaxis="Average Entropy",
    )

    return attention_entropies


def analyze_logits(model, logits, tokens, position=-1, top_k=10):
    """Analyze the model's predictions from logits."""
    print(f"=== Logits Analysis (Position {position}) ===\n")

    # Get logits for the specified position (default: last position)
    position_logits = logits[0, position]  # [vocab_size]

    # Convert to probabilities
    import torch.nn.functional as F

    probabilities = F.softmax(position_logits, dim=-1)

    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    print(f"Top {top_k} predicted tokens:")
    print("Token\t\tProbability\tLogit Score")
    print("-" * 40)

    for i in range(top_k):
        token_idx = top_indices[i].item()
        prob = top_probs[i].item()
        logit_score = position_logits[token_idx].item()
        token_str = model.to_string(token_idx)

        print(f"{token_str:<15}\t{prob:.4f}\t\t{logit_score:.4f}")

    # Show current context
    if position >= 0:
        context = model.to_string(tokens[0, : position + 1])
        print(f"\nContext: {context}")

    print(f"\nLogits shape: {logits.shape}")
    print(f"Vocabulary size: {logits.shape[-1]}")
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

    return top_indices, top_probs


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


def main():
    """Main function demonstrating all activation analysis methods."""
    print("=== TransformerLens Activation Analysis (Main Demo Style) ===\n")

    # Load model
    model = load_model()
    print()

    # Basic activation analysis
    logits, cache, tokens = basic_activation_analysis(model)

    # Analyze logits and predictions
    top_indices, top_probs = analyze_logits(model, logits, tokens)

    # Attention analysis
    head_attention = attention_analysis(model, cache, tokens)

    # Residual stream analysis
    residual_stream_analysis(model, cache, tokens)

    # MLP analysis
    mlp_activations = mlp_analysis(model, cache, tokens)

    # Cross-input comparison
    input_texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "A bird flew over the tree.",
        "Mathematics is the language of science.",
        "Programming requires logical thinking.",
    ]
    activations_dict = cross_input_comparison(model, input_texts)

    # Custom hook example
    logits_orig, logits_zero = custom_hook_example(model, tokens)

    # Induction head test
    induction_head_test(model)

    # Attention head analysis
    attention_entropies = attention_head_analysis(model, cache, tokens)

    # Linear probe demonstration
    probe_results = demonstrate_linear_probes(model)

    print("=== Analysis Complete ===")
    print("\nSummary of methods demonstrated:")
    print("1. Basic activation extraction with cache")
    print("2. Attention pattern analysis and visualization")
    print("3. Residual stream analysis")
    print("4. MLP neuron activation analysis")
    print("5. Cross-input activation comparison")
    print("6. Custom hook interventions")
    print("7. Induction head testing")
    print("8. Attention head entropy analysis")

    print("\nKey Features:")
    print("- Interactive Bokeh/Holoviews visualizations")
    print("- Memory-efficient analysis with torch.no_grad()")
    print("- Comprehensive activation extraction")
    print("- Custom hook interventions")
    print("- Cross-input comparisons")

    print("\nCommon Activation Names:")
    print("- blocks.{layer}.attn.hook_result - Attention output")
    print("- blocks.{layer}.attn.hook_attn_scores - Attention scores")
    print("- blocks.{layer}.mlp.hook_post - MLP output")
    print("- blocks.{layer}.hook_resid_pre - Residual stream before layer")
    print("- blocks.{layer}.hook_resid_post - Residual stream after layer")
    print("- ln_final.hook_normalized - Final layer normalization")


# Linear Probe Functions for Model Analysis


def create_linear_probe(activation_dim, num_classes, device="cpu"):
    """Create a linear probe (classifier) for analyzing activations."""
    probe = torch.nn.Linear(activation_dim, num_classes, bias=True)
    probe.to(device)
    return probe


def extract_activations_for_probe(
    model, texts, labels, layer_idx, activation_name="hook_resid_post"
):
    """Extract activations and labels for training a linear probe."""
    all_activations = []
    all_labels = []

    for text, label in zip(texts, labels):
        # Tokenize and get activations
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Get activations from specified layer
        act_name = get_act_name(activation_name, layer_idx)
        activations = cache[act_name]  # [batch, seq_len, d_model]

        # Use mean pooling over sequence length
        pooled_activations = activations.mean(dim=1)  # [batch, d_model]

        all_activations.append(pooled_activations)
        all_labels.extend([label] * pooled_activations.shape[0])

    # Concatenate all activations
    X = torch.cat(all_activations, dim=0)  # [total_samples, d_model]
    y = torch.tensor(all_labels, dtype=torch.long)

    return X, y


def train_linear_probe(
    probe, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100, device="cpu"
):
    """Train a linear probe on activations."""
    probe.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        probe.train()
        optimizer.zero_grad()

        outputs = probe(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

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

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}: Train Loss = {loss.item():.4f}, "
                f"Val Loss = {val_loss.item():.4f}, Val Acc = {accuracy:.4f}"
            )

    return {
        "train_losses": train_losses,
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


def create_baseline_probes(model, probe_tasks):
    """Create and train baseline probes for various tasks using a HookedTransformer model."""
    print("=== Creating Baseline Linear Probes ===\n")

    probe_results = {}

    for task_name, task_data in probe_tasks.items():
        print(f"Training probe for: {task_name}")

        # Extract activations from different layers
        layer_results = {}

        for layer_idx in range(min(3, model.cfg.n_layers)):  # Test first 3 layers
            print(f"  Layer {layer_idx}...")

            # Extract activations using the provided model
            X, y = extract_activations_for_probe(
                model, task_data["texts"], task_data["labels"], layer_idx
            )

            # Split data
            n_samples = len(X)
            indices = torch.randperm(n_samples)
            split_idx = int(0.8 * n_samples)

            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # Create and train probe
            probe = create_linear_probe(X.shape[1], len(set(y)))
            results = train_linear_probe(probe, X_train, y_train, X_val, y_val)

            # Analyze performance
            performance = analyze_probe_performance(
                probe, X_val, y_val, class_names=task_data.get("class_names")
            )

            layer_results[layer_idx] = {
                "probe": probe,
                "training_results": results,
                "performance": performance,
            }

        probe_results[task_name] = layer_results
        print(f"Completed {task_name}\n")

    return probe_results


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


def get_example_probe_tasks(sample_size=200):
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

    return {
        "math_category": {
            "texts": (
                ds_algebra["train"]["problem"][:sample_size]
                + ds_geometry["train"]["problem"][:sample_size]
                + ds_counting_and_probability["train"]["problem"][:sample_size]
                + ds_intermediate_algebra["train"]["problem"][:sample_size]
                + ds_number_theory["train"]["problem"][:sample_size]
                + ds_prealgebra["train"]["problem"][:sample_size]
                + ds_precalculus["train"]["problem"][:sample_size]
            ),
            "labels": (
                [0] * sample_size  # Counting & Probability
                + [1] * sample_size  # Geometry
                + [2] * sample_size  # Precalculus
                + [3] * sample_size  # Prealgebra
                + [4] * sample_size  # Algebra
                + [5] * sample_size  # Intermediate Algebra
                + [6] * sample_size  # Number Theory
            ),
            "class_names": [
                "Counting & Probability",
                "Precalculus",
                "Geometry",
                "Prealgebra",
                "Algebra",
                "Intermediate Algebra",
                "Number Theory",
            ],
        },
        "math_difficulty": {
            "texts": (
                ds_prealgebra["train"]["problem"][:sample_size]
                + ds_algebra["train"]["problem"][:sample_size]
                + ds_precalculus["train"]["problem"][:sample_size]
                + ds_counting_and_probability["train"]["problem"][:sample_size]
                + ds_intermediate_algebra["train"]["problem"][:sample_size]
                + ds_number_theory["train"]["problem"][:sample_size]
            ),
            "labels": [
                int(level.strip("Level "))
                for level in (
                    ds_prealgebra["train"]["level"][:sample_size]
                    + ds_algebra["train"]["level"][:sample_size]
                    + ds_precalculus["train"]["level"][:sample_size]
                    + ds_counting_and_probability["train"]["level"][:sample_size]
                    + ds_intermediate_algebra["train"]["level"][:sample_size]
                    + ds_number_theory["train"]["level"][:sample_size]
                )
            ],
            "class_names": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
        },
    }


def demonstrate_linear_probes(model):
    """Demonstrate linear probe analysis on a HookedTransformer model."""
    print("=== Linear Probe Demonstration ===\n")

    # Get example probe tasks
    probe_tasks = get_example_probe_tasks()

    # Create and train baseline probes
    probe_results = create_baseline_probes(model, probe_tasks)

    # Visualize results
    visualize_probe_results(probe_results)

    return probe_results


if __name__ == "__main__":
    main()
