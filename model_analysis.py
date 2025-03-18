#!/usr/bin/env python
"""
Weight Watcher evaluation script for genomic BERT models with ALiBi attention.

This script performs a comprehensive analysis of neural network weight matrices
to predict generalization performance without requiring validation data.
Especially useful for analyzing ALiBi attention and genomic pattern recognition.

Requirements:
    pip install weightwatcher torch numpy pandas matplotlib seaborn
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import weightwatcher as ww
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variable for storing ALiBi slopes (used in visualizations)
alibi_slopes_global = {}


def load_model(model_path):
    """
    Load a BERT model from the given path.

    Args:
        model_path: Path to the model directory or checkpoint

    Returns:
        The loaded model
    """
    logger.info(f"Loading model from {model_path}")

    try:
        # Try to load using transformers library first
        try:
            from transformers import BertForMaskedLM
            model = BertForMaskedLM.from_pretrained(model_path)
            logger.info("Successfully loaded model using transformers library")
        except Exception as e:
            logger.warning(f"Failed to load with transformers: {e}")
            logger.info("Attempting to load with PyTorch")

            # Fallback to direct PyTorch loading
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                model = torch.load(model_file, map_location="cpu")
                logger.info(f"Loaded model from {model_file}")
            else:
                # Try to find any .bin or .pt file
                model_files = list(Path(model_path).glob("*.bin")) + list(Path(model_path).glob("*.pt"))
                if model_files:
                    model = torch.load(model_files[0], map_location="cpu")
                    logger.info(f"Loaded model from {model_files[0]}")
                else:
                    raise FileNotFoundError(f"No model files found in {model_path}")

        # Move to CPU for analysis and set to eval mode
        model = model.cpu().eval()
        logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def is_alibi_model(model):
    """
    Check if the model uses ALiBi attention.

    Args:
        model: The model to check

    Returns:
        Boolean indicating if the model uses ALiBi
    """
    # Check config first
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_alibi') and model.config.use_alibi:
            return True
        if hasattr(model.config, 'position_embedding_type') and model.config.position_embedding_type == 'alibi':
            return True
        if hasattr(model.config, 'attention_type') and model.config.attention_type in ['alibi', 'sdpa']:
            return True

    # Check layers
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
        # Check first layer's attention
        layer = model.bert.encoder.layer[0]
        if hasattr(layer.attention.self, 'slopes'):
            return True

    return False


def extract_bert_layer_weights(model):
    """
    Extract weights from BERT model layers.

    Args:
        model: BERT model

    Returns:
        Dictionary of weight matrices and ALiBi slopes
    """
    weights = {}
    alibi_slopes = {}

    # Check if it's a BERT model
    if not hasattr(model, 'bert') or not hasattr(model.bert, 'encoder') or not hasattr(model.bert.encoder, 'layer'):
        logger.warning("Model doesn't have expected BERT structure. Trying alternate extraction method.")

        # Try extracting from state_dict directly
        if hasattr(model, 'state_dict'):
            try:
                state_dict = model.state_dict()
                logger.info(f"Found {len(state_dict)} parameters in state_dict")

                # Collect layer parameters
                for name, param in state_dict.items():
                    if 'weight' in name and param.dim() >= 2:
                        # Extract layer info from parameter name
                        parts = name.split('.')
                        if 'encoder.layer' in name:
                            # Try to extract layer number
                            layer_idx = None
                            for i, part in enumerate(parts):
                                if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                                    layer_idx = parts[i + 1]
                                    break

                            if layer_idx is not None:
                                layer_name = f"layer_{layer_idx}"
                                # Extract component name (query, key, value, etc.)
                                component_name = parts[-2] if len(parts) >= 2 else "weight"

                                # Add to weights dictionary
                                if layer_name not in weights:
                                    weights[layer_name] = {}

                                weights[layer_name][component_name] = param.detach().cpu().numpy()
                                logger.debug(
                                    f"Extracted {name} → {layer_name}.{component_name} with shape {param.shape}")

                        # Check for embeddings
                        elif 'embeddings' in name:
                            if 'embeddings' not in weights:
                                weights['embeddings'] = {}
                            component = parts[-2] if len(parts) >= 2 else "embeddings"
                            weights['embeddings'][component] = param.detach().cpu().numpy()

                # Check if any weights were extracted
                if weights:
                    logger.info(f"Extracted weights for {len(weights)} components from state_dict")
                    return weights, alibi_slopes
            except Exception as e:
                logger.warning(f"Failed to extract weights from state_dict: {e}")

        logger.warning("Could not extract model weights. Returning empty dictionaries.")
        return weights, alibi_slopes

    # Extract embeddings - make sure it's a dictionary
    if hasattr(model.bert, 'embeddings') and hasattr(model.bert.embeddings, 'word_embeddings'):
        try:
            weights['embeddings'] = {
                'word_embeddings': model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
            }
            logger.debug(f"Extracted embeddings with shape {weights['embeddings']['word_embeddings'].shape}")
        except Exception as e:
            logger.warning(f"Failed to extract embeddings: {e}")

    # Extract layer weights
    for i, layer in enumerate(model.bert.encoder.layer):
        try:
            layer_weights = {}

            # Attention weights
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                attn = layer.attention.self
                if hasattr(attn, 'query'):
                    layer_weights['query'] = attn.query.weight.detach().cpu().numpy()
                if hasattr(attn, 'key'):
                    layer_weights['key'] = attn.key.weight.detach().cpu().numpy()
                if hasattr(attn, 'value'):
                    layer_weights['value'] = attn.value.weight.detach().cpu().numpy()

                # ALiBi-specific slopes
                if hasattr(attn, 'slopes'):
                    alibi_slopes[f'layer_{i}'] = attn.slopes.detach().cpu().numpy()

            # Output projection
            if hasattr(layer.attention, 'output') and hasattr(layer.attention.output, 'dense'):
                layer_weights['attention_output'] = layer.attention.output.dense.weight.detach().cpu().numpy()

            # Feedforward weights
            if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                layer_weights['intermediate'] = layer.intermediate.dense.weight.detach().cpu().numpy()

            if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
                layer_weights['output'] = layer.output.dense.weight.detach().cpu().numpy()

            # Only add if we extracted some weights
            if layer_weights:
                weights[f'layer_{i}'] = layer_weights
                logger.debug(f"Extracted weights for layer_{i} with {len(layer_weights)} components")
            else:
                logger.warning(f"No weights extracted for layer_{i}")

        except Exception as e:
            logger.warning(f"Failed to extract weights for layer {i}: {e}")

    logger.info(f"Successfully extracted weights for {len(weights)} layers")
    return weights, alibi_slopes


def analyze_weight_matrix(weight_matrix):
    """
    Analyze a single weight matrix with version-compatible approach.

    Args:
        weight_matrix: NumPy array of the weight matrix

    Returns:
        Dictionary with analysis results
    """
    try:
        # Try newer API first (direct WW analyze function)
        results = ww.analyze(weight_matrix)
        return results
    except Exception as e1:
        try:
            # Try older API next (with WW instance)
            analyzer = ww.WeightWatcher()
            results = analyzer.analyze_matrix(weight_matrix)
            return results
        except Exception as e2:
            try:
                # Try another possible API variation
                results = ww.WeightWatcher().analyze(matrices=[weight_matrix])
                if isinstance(results, pd.DataFrame) and not results.empty:
                    # Convert DataFrame to dict-like object
                    return results.iloc[0]
                return results
            except Exception as e3:
                try:
                    # Final fallback: Try to compute power-law exponent manually
                    # This is the most basic computation that should work with any version
                    logger.info("Trying manual power-law computation")
                    from scipy import stats
                    import numpy as np

                    # Get eigenvalues
                    if min(weight_matrix.shape) > 1:
                        # Use SVD to compute singular values
                        s = np.linalg.svd(weight_matrix, compute_uv=False)

                        # Sort in descending order
                        s = np.sort(s)[::-1]

                        # Remove zeros and very small values
                        s = s[s > 1e-10]

                        if len(s) > 5:  # Need enough points for regression
                            # Log-log transform for power-law fit
                            log_s = np.log(s)
                            log_rank = np.log(np.arange(1, len(s) + 1))

                            # Linear regression on log-log data
                            slope, intercept, r_value, p_value, std_err = stats.linregress(log_rank, log_s)

                            # The power-law exponent is the negative slope
                            alpha = -slope

                            logger.info(f"Manually computed alpha: {alpha:.3f}, r²: {r_value ** 2:.3f}")

                            return {
                                "alpha": alpha,
                                "mp_fit": r_value ** 2,  # Use r² as a proxy for MP fit
                                "kappa": np.nan,
                                "pc_ratio": np.nan
                            }

                    # Fall through to default if above fails
                    return {
                        "alpha": np.nan,
                        "mp_fit": np.nan,
                        "kappa": np.nan,
                        "pc_ratio": np.nan
                    }

                except Exception as e4:
                    logger.warning(f"All analysis methods failed, including manual computation: {e4}")
                    # Return dummy results with NaN values
                    return {
                        "alpha": np.nan,
                        "mp_fit": np.nan,
                        "kappa": np.nan,
                        "pc_ratio": np.nan
                    }


def analyze_layer_weights(weights, output_dir):
    """
    Analyze layer weights using Weight Watcher.

    Args:
        weights: Dictionary of weight matrices
        output_dir: Directory to save results

    Returns:
        DataFrame with analysis results
    """
    logger.info("Analyzing layer weights with Weight Watcher")

    results = []

    # Validate weights is a dictionary
    if not isinstance(weights, dict):
        logger.error(f"Expected weights to be a dictionary, got {type(weights)}")
        return pd.DataFrame()

    for layer_name, layer_weights in tqdm(weights.items(), desc="Analyzing layers"):
        layer_results = {"layer": layer_name}

        # Check if layer_weights is a dictionary - if not, convert it
        if not isinstance(layer_weights, dict):
            logger.warning(
                f"Layer {layer_name} weights is not a dictionary (type: {type(layer_weights)}), converting...")
            # Convert to a dictionary with a single entry
            if isinstance(layer_weights, np.ndarray):
                layer_weights = {'matrix': layer_weights}
            else:
                # Skip this layer if we can't convert it
                logger.warning(f"Cannot convert layer {layer_name} weights to dictionary, skipping.")
                continue

        for weight_name, weight_matrix in layer_weights.items():
            try:
                # Skip if not a numpy array
                if not isinstance(weight_matrix, np.ndarray):
                    logger.warning(f"Weight matrix {layer_name}.{weight_name} is not a numpy array, skipping.")
                    continue

                # Skip if matrix is too small
                if weight_matrix.ndim < 2 or min(weight_matrix.shape) <= 1:
                    logger.warning(
                        f"Weight matrix {layer_name}.{weight_name} is too small: {weight_matrix.shape}, skipping.")
                    continue

                # Use our version-compatible analysis function
                analysis = analyze_weight_matrix(weight_matrix)

                # Add log norm calculation
                layer_results[f"{weight_name}_log_norm"] = np.log(np.linalg.norm(weight_matrix))

                # Add analysis metrics to results
                if isinstance(analysis, dict):
                    # Dictionary-like result
                    for metric, value in analysis.items():
                        if metric in ['alpha', 'mp_fit', 'kappa', 'pc_ratio']:
                            layer_results[f"{weight_name}_{metric}"] = value
                else:
                    # If we got back an object with attributes
                    if hasattr(analysis, 'alpha'):
                        layer_results[f"{weight_name}_alpha"] = analysis.alpha
                    if hasattr(analysis, 'mp_fit'):
                        layer_results[f"{weight_name}_mp_fit"] = analysis.mp_fit
                    if hasattr(analysis, 'kappa'):
                        layer_results[f"{weight_name}_kappa"] = analysis.kappa
                    if hasattr(analysis, 'pc_ratio'):
                        layer_results[f"{weight_name}_pc_ratio"] = analysis.pc_ratio

            except Exception as e:
                logger.warning(f"Failed to analyze {layer_name}.{weight_name}: {e}")
                # Set failed metrics to NaN
                layer_results[f"{weight_name}_alpha"] = np.nan
                layer_results[f"{weight_name}_log_norm"] = np.nan

        results.append(layer_results)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "layer_analysis.csv"), index=False)

    return results_df


def analyze_alibi_slopes(slopes, output_dir):
    """
    Analyze ALiBi slopes distribution.

    Args:
        slopes: Dictionary of ALiBi slope arrays
        output_dir: Directory to save results

    Returns:
        DataFrame with analysis results
    """
    if not slopes:
        logger.info("No ALiBi slopes found")
        return None

    logger.info("Analyzing ALiBi slopes")

    # Prepare results
    results = []

    for layer_name, slope_values in slopes.items():
        results.append({
            "layer": layer_name,
            "min_slope": np.min(slope_values),
            "max_slope": np.max(slope_values),
            "mean_slope": np.mean(slope_values),
            "median_slope": np.median(slope_values),
            "std_slope": np.std(slope_values)
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(os.path.join(output_dir, "alibi_slopes.csv"), index=False)

    return results_df


def run_full_model_analysis(model, output_dir):
    """
    Run full model analysis using WeightWatcher with version compatibility.

    Args:
        model: PyTorch model
        output_dir: Directory to save results

    Returns:
        DataFrame with analysis results or None if failed
    """
    try:
        logger.info("Running full model analysis with Weight Watcher")

        # Try different API versions
        try:
            # Version-specific approaches

            # Try newer API first
            analyzer = ww.WeightWatcher(model=model)
            ww_results = analyzer.analyze()

            # Try to generate a summary
            try:
                if hasattr(analyzer, 'get_summary'):
                    summary = analyzer.get_summary()
                    with open(os.path.join(output_dir, "ww_summary.txt"), "w") as f:
                        f.write(str(summary))
            except Exception as e:
                logger.warning(f"Failed to generate summary: {e}")

            # Try to generate plots - different method names in different versions
            try:
                for plot_method in ['plot', 'plot_details', 'details_plot']:
                    if hasattr(analyzer, plot_method):
                        plot_func = getattr(analyzer, plot_method)
                        try:
                            fig = plot_func()
                            fig.savefig(os.path.join(output_dir, f"ww_{plot_method}.png"), dpi=300)
                            plt.close(fig)
                            logger.info(f"Created plot using {plot_method}")
                            break
                        except Exception as plot_err:
                            logger.warning(f"Failed with {plot_method}: {plot_err}")
            except Exception as e:
                logger.warning(f"All plotting methods failed: {e}")

            return ww_results

        except Exception as e:
            logger.warning(f"First API approach failed: {e}")

            # Try older API
            try:
                analyzer = ww.WeightWatcher()
                ww_results = analyzer.analyze_model(model)

                # Save results
                if isinstance(ww_results, pd.DataFrame):
                    ww_results.to_csv(os.path.join(output_dir, "full_model_analysis.csv"))
                else:
                    with open(os.path.join(output_dir, "full_model_analysis.txt"), "w") as f:
                        f.write(str(ww_results))

                return ww_results

            except Exception as e2:
                logger.warning(f"Second API approach failed: {e2}")

                # Final attempt - get state_dict and analyze each matrix
                try:
                    logger.info("Attempting direct analysis of model weights")
                    state_dict = model.state_dict()
                    matrices = []
                    matrix_names = []

                    for name, param in state_dict.items():
                        if 'weight' in name and param.dim() >= 2:
                            matrices.append(param.detach().cpu().numpy())
                            matrix_names.append(name)

                    # Analyze with our most generic approach
                    results = []
                    for i, matrix in enumerate(matrices):
                        result = analyze_weight_matrix(matrix)
                        if isinstance(result, dict):
                            result['name'] = matrix_names[i]
                            results.append(result)

                    if results:
                        ww_results = pd.DataFrame(results)
                        ww_results.to_csv(os.path.join(output_dir, "full_model_analysis.csv"))
                        return ww_results

                except Exception as e3:
                    logger.warning(f"All API approaches failed: {e3}")

                    # Ultimate fallback: Try direct SVD analysis
                    try:
                        logger.info("Attempting direct SVD analysis")
                        from scipy import stats
                        import numpy as np

                        results = []
                        state_dict = model.state_dict()

                        for name, param in state_dict.items():
                            if 'weight' in name and param.dim() >= 2:
                                weight = param.detach().cpu().numpy()

                                if min(weight.shape) > 5:  # Need enough dimensions
                                    # Get singular values
                                    s = np.linalg.svd(weight, compute_uv=False)

                                    # Sort in descending order
                                    s = np.sort(s)[::-1]

                                    # Remove zeros and very small values
                                    s = s[s > 1e-10]

                                    if len(s) > 5:  # Need enough points for regression
                                        # Log-log transform for power-law fit
                                        log_s = np.log(s)
                                        log_rank = np.log(np.arange(1, len(s) + 1))

                                        # Linear regression on log-log data
                                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_rank, log_s)

                                        # The power-law exponent is the negative slope
                                        alpha = -slope

                                        results.append({
                                            'name': name,
                                            'alpha': alpha,
                                            'r_squared': r_value ** 2,
                                            'shape': str(weight.shape),
                                            'norm': np.linalg.norm(weight)
                                        })

                        if results:
                            # Convert to DataFrame
                            svd_df = pd.DataFrame(results)
                            svd_df.to_csv(os.path.join(output_dir, "svd_analysis.csv"))

                            # Create a simple visualization
                            plt.figure(figsize=(12, 8))
                            plt.scatter(svd_df['alpha'], np.log(svd_df['norm']))
                            plt.xlabel('Power-Law Exponent (α)')
                            plt.ylabel('Log Weight Norm')
                            plt.title('SVD-Based Power-Law Analysis')
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, "svd_analysis.png"), dpi=300)
                            plt.close()

                            logger.info(f"SVD analysis complete with {len(results)} matrices")
                            return svd_df

                    except Exception as e4:
                        logger.warning(f"SVD analysis failed: {e4}")

                    return None

    except Exception as e:
        logger.warning(f"Full model analysis failed: {e}")
        return None


def generate_visualizations(layer_results, slope_results, output_dir):
    """
    Generate visualizations for Weight Watcher results.

    Args:
        layer_results: DataFrame with layer analysis results
        slope_results: DataFrame with ALiBi slope analysis
        output_dir: Directory to save visualizations
    """
    logger.info("Generating visualizations")

    os.makedirs(output_dir, exist_ok=True)

    # Check if layer_results is valid
    if layer_results is None or layer_results.empty:
        logger.warning("No layer results available for visualization")
        return

    # Make sure 'layer' column exists
    if 'layer' not in layer_results.columns:
        logger.warning("'layer' column missing from results, skipping visualizations")
        return

    # 1. Alpha values heatmap
    try:
        alpha_columns = [col for col in layer_results.columns if
                         col.endswith('_alpha') and not layer_results[col].isnull().all()]

        if alpha_columns:
            plt.figure(figsize=(16, 10))

            # Extract layer numbers or names
            layers = layer_results['layer'].tolist()

            # Create heatmap data
            heatmap_data = layer_results[alpha_columns].copy()

            # Replace column names for better display
            heatmap_data.columns = [col.replace('_alpha', '') for col in alpha_columns]

            # Create heatmap
            ax = sns.heatmap(heatmap_data.set_index('layer').T,
                             annot=True,
                             cmap='viridis',
                             fmt=".2f",
                             linewidths=.5)

            plt.title("Power-Law Exponents (α) Across Layers")
            plt.ylabel("Weight Matrix")
            plt.xlabel("Layer")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "alpha_heatmap.png"), dpi=300)
            plt.close()
            logger.info(f"Created alpha heatmap: {os.path.join(output_dir, 'alpha_heatmap.png')}")
    except Exception as e:
        logger.warning(f"Failed to create alpha heatmap: {e}")

    # 2. Alpha vs. Weight Norm scatter plot
    try:
        # Check if we have norm columns
        norm_columns = [col for col in layer_results.columns if col.endswith('_log_norm')]
        alpha_columns = [col for col in layer_results.columns if col.endswith('_alpha')]

        if norm_columns and alpha_columns:
            plt.figure(figsize=(12, 8))

            # Get component types
            component_types = set([col.split('_')[0] for col in alpha_columns])

            # Plot each component type with a different color
            for component in component_types:
                alpha_col = f"{component}_alpha"
                norm_col = f"{component}_log_norm"

                if alpha_col in layer_results.columns and norm_col in layer_results.columns:
                    # Make sure we have non-null values
                    valid_mask = ~(layer_results[alpha_col].isnull() | layer_results[norm_col].isnull())
                    if valid_mask.any():
                        plt.scatter(
                            layer_results.loc[valid_mask, alpha_col],
                            layer_results.loc[valid_mask, norm_col],
                            label=component,
                            alpha=0.7
                        )

            plt.xlabel("Power-Law Exponent (α)")
            plt.ylabel("Log Weight Norm")
            plt.title("Relationship Between Power-Law Behavior and Weight Magnitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "alpha_vs_norm.png"), dpi=300)
            plt.close()
            logger.info(f"Created alpha vs. norm plot: {os.path.join(output_dir, 'alpha_vs_norm.png')}")
    except Exception as e:
        logger.warning(f"Failed to create alpha vs. norm plot: {e}")

    # 3. ALiBi slopes visualizations
    if slope_results is not None and not slope_results.empty:
        try:
            plt.figure(figsize=(12, 6))
            plt.bar(slope_results['layer'], slope_results['mean_slope'], yerr=slope_results['std_slope'])
            plt.title("ALiBi Slopes Across Layers")
            plt.xlabel("Layer")
            plt.ylabel("Average Slope Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "alibi_slopes.png"), dpi=300)
            plt.close()
            logger.info(f"Created ALiBi slopes plot: {os.path.join(output_dir, 'alibi_slopes.png')}")

            # Check if alibi_slopes_global is defined before using it
            if 'alibi_slopes_global' in globals() and alibi_slopes_global:
                try:
                    # Slope distribution across all layers
                    all_slopes = np.concatenate([slopes for layer, slopes in alibi_slopes_global.items()])
                    plt.figure(figsize=(10, 6))
                    sns.histplot(all_slopes, kde=True)
                    plt.title("Distribution of ALiBi Slopes Across All Layers")
                    plt.xlabel("Slope Value")
                    plt.ylabel("Frequency")
                    plt.savefig(os.path.join(output_dir, "alibi_slope_distribution.png"), dpi=300)
                    plt.close()
                    logger.info(
                        f"Created ALiBi slope distribution plot: {os.path.join(output_dir, 'alibi_slope_distribution.png')}")
                except Exception as e:
                    logger.warning(f"Failed to create ALiBi slope distribution plot: {e}")
        except Exception as e:
            logger.warning(f"Failed to create ALiBi visualizations: {e}")


def generate_interpretation(layer_results, slope_results, is_alibi, output_dir):
    """
    Generate interpretation of Weight Watcher results.

    Args:
        layer_results: DataFrame with layer analysis results
        slope_results: DataFrame with ALiBi slope analysis
        is_alibi: Boolean indicating if the model uses ALiBi
        output_dir: Directory to save interpretation
    """
    logger.info("Generating interpretation")

    # Define optimal alpha value ranges for sequence models
    optimal_ranges = {
        'min': 2.0,  # Values below suggest poor generalization
        'good': 2.5,  # Target value for good generalization
        'max': 3.5  # Values above suggest memorization
    }

    # Check if layer_results is valid
    if layer_results is None or layer_results.empty:
        logger.warning("No layer results available for interpretation")
        with open(os.path.join(output_dir, "interpretation.txt"), "w") as f:
            f.write("WEIGHT WATCHER ANALYSIS INTERPRETATION\n")
            f.write("=====================================\n\n")
            f.write("No layer results available for interpretation.\n")
            f.write("The analysis was not able to compute metrics for this model.\n")
        return

    # Extract alpha values for attention and feedforward components
    alpha_cols = [col for col in layer_results.columns if col.endswith('_alpha')]
    attention_alphas = [col for col in alpha_cols if any(x in col for x in ['query', 'key', 'value', 'attention'])]
    ff_alphas = [col for col in alpha_cols if any(x in col for x in ['intermediate', 'output'])]

    attention_avg = layer_results[attention_alphas].mean().mean() if attention_alphas else np.nan
    ff_avg = layer_results[ff_alphas].mean().mean() if ff_alphas else np.nan
    overall_avg = layer_results[alpha_cols].mean().mean() if alpha_cols else np.nan

    # Generate interpretation report
    with open(os.path.join(output_dir, "interpretation.txt"), "w") as f:
        f.write("WEIGHT WATCHER ANALYSIS INTERPRETATION\n")
        f.write("=====================================\n\n")

        f.write("OVERALL MODEL ASSESSMENT:\n")
        f.write(f"Average Power-Law Exponent (α): {overall_avg:.3f}\n")
        f.write(f"  - Attention Components α: {attention_avg:.3f}\n")
        f.write(f"  - Feedforward Components α: {ff_avg:.3f}\n\n")

        # Interpretation of alpha values
        if not np.isnan(overall_avg):
            if overall_avg < optimal_ranges['min']:
                f.write("ALERT: Model shows signs of UNDERFITTING (α too low)\n")
                f.write("- May struggle to capture complex genomic patterns\n")
                f.write("- Consider longer training or larger model\n")
            elif overall_avg > optimal_ranges['max']:
                f.write("ALERT: Model shows signs of OVERFITTING (α too high)\n")
                f.write("- May memorize training data rather than generalize\n")
                f.write("- Consider more regularization or reducing model size\n")
            else:
                f.write("GOOD: Model appears well-balanced for generalization\n")

        # Difference between attention and feedforward components
        if not np.isnan(attention_avg) and not np.isnan(ff_avg):
            diff = abs(attention_avg - ff_avg)
            if diff > 0.5:
                f.write(f"\nNOTE: Large difference ({diff:.2f}) between attention and feedforward components\n")
                if attention_avg > ff_avg:
                    f.write("- Attention layers appear to be stronger than feedforward layers\n")
                    f.write("- Model may focus more on sequence relationships than feature extraction\n")
                else:
                    f.write("- Feedforward layers appear stronger than attention layers\n")
                    f.write("- Model may focus more on feature extraction than sequence relationships\n")

        # Layer-specific insights
        f.write("\nLAYER-SPECIFIC INSIGHTS:\n")
        for i, row in layer_results.iterrows():
            layer_name = row['layer']
            layer_alphas = [v for k, v in row.items() if k.endswith('_alpha') and not np.isnan(v)]

            if not layer_alphas:
                continue

            avg_layer_alpha = sum(layer_alphas) / len(layer_alphas)

            f.write(f"Layer {layer_name}: Avg α = {avg_layer_alpha:.3f} - ")

            if avg_layer_alpha < optimal_ranges['min']:
                f.write("WEAK - This layer may be underutilized\n")
            elif avg_layer_alpha > optimal_ranges['max']:
                f.write("OVERSPECIALIZED - This layer may be memorizing\n")
            else:
                f.write("HEALTHY - This layer shows good generalization properties\n")

        # ALiBi-specific analysis
        if is_alibi and slope_results is not None and not slope_results.empty:
            f.write("\nALiBi ATTENTION ANALYSIS:\n")

            avg_slope = slope_results['mean_slope'].mean()
            avg_std = slope_results['std_slope'].mean()

            f.write(f"Average ALiBi slope: {avg_slope:.4f}\n")
            f.write(f"Average slope variation: {avg_std:.4f}\n\n")

            if avg_std > 0.5:
                f.write("HIGH SLOPE VARIATION: Model has diverse attention patterns across heads\n")
                f.write("- Good for handling different types of genomic sequences\n")
                f.write("- Expect good performance on varied sequence lengths\n")
            else:
                f.write("LOW SLOPE VARIATION: Model has similar attention patterns across heads\n")
                f.write("- May struggle with diverse genomic sequences\n")
                f.write("- Consider modifying attention head initialization\n")

        f.write("\nGENOVMIC BERT-SPECIFIC CONSIDERATIONS:\n")
        f.write("- Power-law behavior is crucial for genome sequence generalization\n")
        f.write("- ALiBi attention requires specific weight structure for extrapolation\n")
        f.write("- Genomic models benefit from careful balance between memorization and generalization\n")

    # Generate actionable recommendations
    with open(os.path.join(output_dir, "recommendations.txt"), "w") as f:
        f.write("ACTIONABLE RECOMMENDATIONS\n")
        f.write("=========================\n\n")

        if not np.isnan(overall_avg):
            if overall_avg < optimal_ranges['min']:
                f.write("1. INCREASE TRAINING TIME: Your model appears to be underfitting\n")
                f.write("2. CHECK DATA QUALITY: Ensure genomic sequences are properly preprocessed\n")
                f.write("3. INCREASE MODEL CAPACITY: Consider more attention heads or larger hidden size\n")
            elif overall_avg > optimal_ranges['max']:
                f.write("1. ADD REGULARIZATION: Try adding dropout to attention layers\n")
                f.write("2. REDUCE MODEL SIZE: Your model may be too complex for your dataset\n")
                f.write("3. ADD MORE TRAINING DATA: Diverse genomic examples will help generalization\n")
            else:
                f.write("1. MAINTAIN CURRENT ARCHITECTURE: Your model shows good balance\n")
                f.write("2. FINE-TUNE SELECT LAYERS: Focus on layers with extreme α values\n")
                f.write("3. CONSIDER PRUNING: Some attention heads may be redundant\n")

        # Add genomic-specific recommendations
        f.write("\nGENOMIC-SPECIFIC RECOMMENDATIONS:\n")
        f.write("1. BENCHMARK ON VARIABLE LENGTH SEQUENCES: Test extrapolation capabilities\n")
        f.write("2. EVALUATE ON REVERSE COMPLEMENT SEQUENCES: Check biological consistency\n")
        f.write("3. TEST WITH AMBIGUOUS NUCLEOTIDES: Ensure robustness to sequence variations\n")

        if is_alibi:
            f.write("\nALiBi-SPECIFIC RECOMMENDATIONS:\n")
            f.write("1. VERIFY EXTRAPOLATION: Test on sequences 2-4x longer than training length\n")
            f.write("2. CHECK ATTENTION SLOPES: Ensure proper distribution for genomic data\n")
            f.write("3. CONSIDER ALIBI TUNING: Adjust slope initialization based on sequence properties\n")


def run_ww_analysis(model, output_dir):
    """
    Run full Weight Watcher analysis on the model.

    Args:
        model: Model to analyze
        output_dir: Directory to save results
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Check if the model uses ALiBi attention
        is_alibi_model_flag = is_alibi_model(model)
        logger.info(f"Model uses ALiBi attention: {is_alibi_model_flag}")

        # 1. Extract weights from model
        layer_weights, alibi_slopes = extract_bert_layer_weights(model)
        logger.info(f"Extracted weights from {len(layer_weights)} layers")

        # Store alibi_slopes in global scope for visualization
        global alibi_slopes_global
        alibi_slopes_global = alibi_slopes

        # 2. Analyze layer weights
        layer_results = analyze_layer_weights(layer_weights, output_dir)

        # 3. Analyze ALiBi slopes if present
        slope_results = None
        if alibi_slopes:
            try:
                slope_results = analyze_alibi_slopes(alibi_slopes, output_dir)
            except Exception as e:
                logger.warning(f"Failed to analyze ALiBi slopes: {e}")

        # 4. Run full model analysis using Weight Watcher
        run_full_model_analysis(model, output_dir)

        # 5. Generate visualizations
        try:
            generate_visualizations(layer_results, slope_results, output_dir)
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

        # 6. Generate interpretation and recommendations
        try:
            generate_interpretation(layer_results, slope_results, is_alibi_model_flag, output_dir)
        except Exception as e:
            logger.warning(f"Failed to generate interpretation: {e}")

        logger.info(f"Analysis complete. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Weight Watcher evaluation for genomic BERT models")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--output_dir", type=str, default="./ww_analysis",
                        help="Directory to save Weight Watcher analysis results")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    logger.info("=" * 80)
    logger.info("GENOMIC BERT WEIGHT WATCHER EVALUATION".center(80))
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Load model
        model = load_model(args.model_path)

        # Run analysis
        run_ww_analysis(model, args.output_dir)

        # Print completion message
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE".center(80))
        logger.info(f"Results saved to {args.output_dir}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Stack trace:")
        return 1


if __name__ == "__main__":
    sys.exit(main())