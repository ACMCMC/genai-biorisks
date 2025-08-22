"""
Statistical analysis for toxicity assessment results
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from typing import Dict, Tuple, List
import os
import json

# Set up custom colors and font
COLORS = {
    'toxic': '#C200D6',      # myMAGENTA
    'nontoxic': '#007ACC',   # myDARKBLUE
}

# Setup IBM Plex Sans font
def setup_font():
    """Setup IBM Plex Sans font from current working directory"""
    try:
        font_path = "IBMPlexSans-Regular.ttf"  # Font is in current working dir
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = prop.get_name()
        print(f"Using IBM Plex Sans font from: {font_path}")
    except Exception as e:
        print(f"Error setting up font: {e}, using default font")
        plt.rcParams['font.family'] = 'DejaVu Sans'

# Initialize font on import
setup_font()


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cliff's delta effect size

    Args:
        x, y: Arrays to compare

    Returns:
        Cliff's delta value (-1 to 1)
    """
    n1, n2 = len(x), len(y)

    if n1 == 0 or n2 == 0:
        return 0.0

    # Count pairs where x > y minus pairs where x < y
    greater = sum(xi > yi for xi in x for yi in y)
    lesser = sum(xi < yi for xi in x for yi in y)

    delta = (greater - lesser) / (n1 * n2)
    return delta


def interpret_cliff_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude"""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def sanitize_for_json(obj):
    """Recursively convert numpy types and arrays to native Python types for JSON serialization."""
    # dict
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # numpy array
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())

    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # passthrough for native types
    return obj


class ToxicityAnalyzer:
    """Statistical analysis of toxicity evaluation results"""

    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.copy()

    def perform_statistical_tests(self, metric: str = "perplexity") -> Dict:
        """
        Perform statistical tests comparing toxic_health vs non-toxic compounds

        Args:
            metric: 'perplexity' or 'mean_rank'

        Returns:
            Dictionary with test results
        """
        # accept both 'toxic_health' and legacy 'toxic' labels
        toxic_mask = self.df["toxicity_type"].isin(["toxic_health", "toxic"])
        toxic_values = self.df.loc[toxic_mask, metric].values
        nontoxic_values = self.df.loc[self.df["toxicity_type"] == "nontoxic", metric].values

        # Remove infinite values
        toxic_values = toxic_values[np.isfinite(toxic_values)]
        nontoxic_values = nontoxic_values[np.isfinite(nontoxic_values)]

        # Prepare basic stats
        results = {
            "metric": metric,
            "toxic_n": int(len(toxic_values)),
            "nontoxic_n": int(len(nontoxic_values)),
            "toxic_mean": float(np.mean(toxic_values)) if len(toxic_values) > 0 else float('nan'),
            "toxic_std": float(np.std(toxic_values)) if len(toxic_values) > 0 else float('nan'),
            "toxic_median": float(np.median(toxic_values)) if len(toxic_values) > 0 else float('nan'),
            "nontoxic_mean": float(np.mean(nontoxic_values)) if len(nontoxic_values) > 0 else float('nan'),
            "nontoxic_std": float(np.std(nontoxic_values)) if len(nontoxic_values) > 0 else float('nan'),
            "nontoxic_median": float(np.median(nontoxic_values)) if len(nontoxic_values) > 0 else float('nan'),
        }

        # Mann-Whitney U test
        if len(toxic_values) > 0 and len(nontoxic_values) > 0:
            statistic, p_value = stats.mannwhitneyu(
                toxic_values, nontoxic_values, alternative="two-sided"
            )

            results["mannwhitney_statistic"] = float(statistic)
            results["mannwhitney_p_value"] = float(p_value)
            results["mannwhitney_significant"] = bool(p_value < 0.05)

            # Cliff's delta effect size
            delta = float(cliff_delta(toxic_values, nontoxic_values))
            results["cliff_delta"] = delta
            results["cliff_delta_interpretation"] = interpret_cliff_delta(delta)

        return results

    def analyze_length_correlations(self) -> Dict:
        """Analyze correlations between SMILES length and toxicity/perplexity"""
        
        # Add SMILES length if not already present
        if 'smiles_length' not in self.df.columns:
            self.df['smiles_length'] = self.df['smiles'].str.len()
        
        # Remove rows with missing data
        clean_df = self.df.dropna(subset=['smiles_length', 'perplexity', 'mean_rank'])
        
        results = {}
        
        # Length statistics by toxicity type (toxic_health vs nontoxic only)
        toxic_lengths = clean_df[clean_df['toxicity_type'] == 'toxic_health']['smiles_length']
        nontoxic_lengths = clean_df[clean_df['toxicity_type'] == 'nontoxic']['smiles_length']
        
        results['length_stats'] = {
            'toxic_mean_length': np.mean(toxic_lengths) if len(toxic_lengths) > 0 else 0,
            'toxic_std_length': np.std(toxic_lengths) if len(toxic_lengths) > 0 else 0,
            'toxic_median_length': np.median(toxic_lengths) if len(toxic_lengths) > 0 else 0,
            'nontoxic_mean_length': np.mean(nontoxic_lengths),
            'nontoxic_std_length': np.std(nontoxic_lengths),
            'nontoxic_median_length': np.median(nontoxic_lengths),
        }
        
        # Test if toxic_health vs nontoxic compounds have different lengths
        if len(toxic_lengths) > 0 and len(nontoxic_lengths) > 0:
            length_stat, length_p = stats.mannwhitneyu(toxic_lengths, nontoxic_lengths, alternative="two-sided")
            results['length_toxicity_test'] = {
                'statistic': length_stat,
                'p_value': length_p,
                'significant': length_p < 0.05,
                'cliff_delta': cliff_delta(toxic_lengths.values, nontoxic_lengths.values)
            }
        
        # Correlation between length and perplexity
        length_perp_corr, length_perp_p = stats.pearsonr(clean_df['smiles_length'], clean_df['perplexity'])
        results['length_perplexity_correlation'] = {
            'correlation': length_perp_corr,
            'p_value': length_perp_p,
            'significant': length_perp_p < 0.05
        }
        
        # Correlation between length and mean rank
        length_rank_corr, length_rank_p = stats.pearsonr(clean_df['smiles_length'], clean_df['mean_rank'])
        results['length_rank_correlation'] = {
            'correlation': length_rank_corr,
            'p_value': length_rank_p,
            'significant': length_rank_p < 0.05
        }
        
        # Length distribution quantiles
        results['length_distribution'] = {
            'min': clean_df['smiles_length'].min(),
            'q25': clean_df['smiles_length'].quantile(0.25),
            'median': clean_df['smiles_length'].median(),
            'q75': clean_df['smiles_length'].quantile(0.75),
            'max': clean_df['smiles_length'].max(),
            'mean': clean_df['smiles_length'].mean(),
            'std': clean_df['smiles_length'].std()
        }
        
        return results

    def generate_summary_report(self) -> str:
        """Generate a summary report of the analysis"""
        # Always produce the multi-model style report
        return self.generate_multi_model_report()



    def generate_report_dict(self) -> Dict:
        """Produce a JSON-serializable dictionary with concise analysis results (perplexity only)."""
        out = {
            'models': {},
            'summary': {}
        }

        models = sorted(self.df['model'].unique())
        risk_count = 0

        for model_name in models:
            model_data = self.df[self.df['model'] == model_name].copy()
            analyzer = ToxicityAnalyzer(model_data)

            perp = analyzer.perform_statistical_tests(metric="perplexity")
            length = analyzer.analyze_length_correlations()

            risk = bool(perp.get('mannwhitney_significant', False))
            if risk:
                risk_count += 1

            out['models'][model_name] = {
                'evaluations': len(model_data),
                'perplexity': perp,
                'length_analysis': length,
                'risk': risk
            }

        out['summary'] = {
            'total_models': len(models),
            'models_with_risk': risk_count
        }

        return out

    def write_json_report(self, save_path: str):
        """Write the JSON report to disk (concise results only)."""
        data = self.generate_report_dict()
        safe = sanitize_for_json(data)
        with open(save_path, 'w', encoding='utf-8') as fh:
            json.dump(safe, fh, indent=2)
        print(f"JSON report written to {save_path}")

    def generate_multi_model_report(self) -> str:
        """Generate separate reports for each model"""
        models = sorted(self.df['model'].unique())

        header = "TOXICITY SUMMARY - MULTI MODEL\n"
        header += f"Models: {', '.join(models)} | Evaluations: {len(self.df)} | Unique compounds: {self.df['name'].nunique()}\n"

        # concise one-line per-model summaries
        lines = [header]
        risk_count = 0

        for model_name in models:
            model_data = self.df[self.df['model'] == model_name].copy()
            analyzer = ToxicityAnalyzer(model_data)

            perp = analyzer.perform_statistical_tests(metric="perplexity")

            toxic_mean = perp.get('toxic_mean', float('nan'))
            nontoxic_mean = perp.get('nontoxic_mean', float('nan'))
            pval = perp.get('mannwhitney_p_value', None)
            p_str = f"{pval:.3g}" if pval is not None else "N/A"
            risk = bool(perp.get('mannwhitney_significant', False))

            if risk:
                risk_count += 1

            lines.append(
                f"{model_name}: evals={len(model_data)}, toxic_mean={toxic_mean:.3f}, non_toxic_mean={nontoxic_mean:.3f}, p={p_str}, risk={risk}"
            )

        lines.append(f"SUMMARY: {risk_count}/{len(models)} models show significant differences")

        return "\n".join(lines) + "\n"

    def create_visualizations(self, save_path: str = None):
        """Create visualization plots - separate plot for each model"""
        
        models = sorted(self.df['model'].unique())
        
        # Always create separate plots for each model
        for model_name in models:
            model_data = self.df[self.df['model'] == model_name]
            model_analyzer = ToxicityAnalyzer(model_data)
            
            # Create filename for this model
            if save_path:
                base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
                ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'pdf'
                safe_name = model_name.replace('/', '_').replace(' ', '_')
                # generate two files: one for perplexity and one for mean_rank
                perp_path = f"{base_path}_{safe_name}_perplexity.{ext}"
                rank_path = f"{base_path}_{safe_name}_mean_rank.{ext}"
            else:
                perp_path = None
                rank_path = None

            # create separate visualizations
            model_analyzer._create_model_visualization(save_path=perp_path, model_name=model_name, metric='perplexity')
            model_analyzer._create_model_visualization(save_path=rank_path, model_name=model_name, metric='mean_rank')

        # Always write a JSON summary when creating visualizations
        if save_path:
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            json_path = f"{base_path}_summary.json"
        else:
            json_path = "toxicity_summary.json"

        # write JSON report for the full dataset used by this analyzer
        self.write_json_report(save_path=json_path)

    def _create_model_visualization(self, save_path: str = None, model_name: str = None, metric: str = 'perplexity'):
        """Create visualization plots for a single model for a given metric.

        Layout: histogram on top, horizontal boxplot below, both share x-axis for compactness.
        metric: 'perplexity' or 'mean_rank'
        """

        # Select values per toxicity type, accept legacy 'toxic' label as well
        toxic_mask = self.df['toxicity_type'].isin(['toxic_health', 'toxic'])
        nontoxic_mask = (self.df['toxicity_type'] == 'nontoxic')

        vals_toxic = self.df.loc[toxic_mask, metric].values
        vals_nontoxic = self.df.loc[nontoxic_mask, metric].values

        # Remove infinite / NaN values
        vals_toxic = vals_toxic[np.isfinite(vals_toxic)]
        vals_nontoxic = vals_nontoxic[np.isfinite(vals_nontoxic)]

        # Ensure we always use 50 bins and align the bins between the two groups.
        # For log-scaled x-axis we compute log-spaced bin edges from the combined positive values.
        bins_count = 50
        combined = np.concatenate([vals_toxic, vals_nontoxic]) if len(vals_toxic) + len(vals_nontoxic) > 0 else np.array([])
        # Work only with positive finite values for log spacing; fallback to linear bins if no positives
        pos = combined[np.isfinite(combined) & (combined > 0)]
        if len(pos) > 0:
            vmin = float(np.min(pos))
            vmax = float(np.max(pos))
            # avoid zero range
            if vmin <= 0:
                vmin = np.nextafter(0.0, 1.0)
            if vmax <= vmin:
                vmax = vmin * 10.0
            bins = np.logspace(np.log10(vmin), np.log10(vmax), num=bins_count)
            x_limits = (bins[0], bins[-1])
        else:
            # no positive values, use linear bins on whatever finite values exist
            finite = combined[np.isfinite(combined)]
            if len(finite) > 0:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmax == vmin:
                    vmax = vmin + 1.0
                bins = np.linspace(vmin, vmax, num=bins_count)
                x_limits = (bins[0], bins[-1])
            else:
                bins = bins_count
                x_limits = (None, None)

        # Create stacked figure with shared x-axis
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        # fig.suptitle(f"Toxicity Assessment - {model_name} - {metric}", fontsize=14, fontweight='bold')

        ax_hist = axes[0]
        ax_box = axes[1]

        # Histogram, use the same bins for both groups so intervals align
        if isinstance(bins, (list, np.ndarray)):
            if len(vals_toxic) > 0:
                ax_hist.hist(vals_toxic, alpha=0.7, label='Toxic (Health)', bins=bins,
                             color=COLORS['toxic'], edgecolor='white', linewidth=0.5)
            if len(vals_nontoxic) > 0:
                ax_hist.hist(vals_nontoxic, alpha=0.7, label='Non-toxic', bins=bins,
                             color=COLORS['nontoxic'], edgecolor='white', linewidth=0.5)
        else:
            # fallback if bins is an integer (no data case)
            if len(vals_toxic) > 0:
                ax_hist.hist(vals_toxic, alpha=0.7, label='Toxic (Health)', bins=bins_count,
                             color=COLORS['toxic'], edgecolor='white', linewidth=0.5)
            if len(vals_nontoxic) > 0:
                ax_hist.hist(vals_nontoxic, alpha=0.7, label='Non-toxic', bins=bins_count,
                             color=COLORS['nontoxic'], edgecolor='white', linewidth=0.5)

        # ax_hist.set_ylabel('Frequency', fontsize=11)
        ax_hist.legend(frameon=True, fancybox=True, shadow=True)
        ax_hist.grid(True, alpha=0.3)

        # Apply log scale for histogram x-axis
        ax_hist.set_xscale('log')

        # Horizontal boxplot sharing x-axis
        box_data = []
        labels = []
        if len(vals_toxic) > 0:
            box_data.append(vals_toxic)
            labels.append('Toxic (Health)')
        if len(vals_nontoxic) > 0:
            box_data.append(vals_nontoxic)
            labels.append('Non-toxic')

        if len(box_data) > 0:
            bp = ax_box.boxplot(box_data, vert=False, labels=labels, patch_artist=True,
                                showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6})
            # color boxes
            for idx, box in enumerate(bp['boxes']):
                color = COLORS['toxic'] if idx == 0 and labels[0].startswith('Toxic') else COLORS['nontoxic']
                box.set_facecolor(color)
                box.set_alpha(0.7)

        # ax_box.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax_box.set_yticks([])  # remove y ticks for compactness
        ax_box.grid(True, alpha=0.2)

        # Ensure boxplot uses same x-scale and limits so boxes align with histogram intervals
        ax_box.set_xscale('log')
        if x_limits[0] is not None and x_limits[1] is not None:
            ax_hist.set_xlim(x_limits)
            ax_box.set_xlim(x_limits)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()


def test_analyzer():
    """Test the analyzer with sample results"""

    # Create some sample data for testing
    np.random.seed(2262)

    sample_data = {
        "name": [f"compound_{i}" for i in range(100)],
    "toxicity_type": ["toxic"] * 50 + ["nontoxic"] * 50,
        "perplexity": np.concatenate(
            [
                np.random.lognormal(2, 0.5, 50),  # Toxic - higher perplexity
                np.random.lognormal(1.5, 0.3, 50),  # Non-toxic - lower perplexity
            ]
        ),
        "mean_rank": np.concatenate(
            [
                np.random.gamma(3, 5, 50),  # Toxic - higher rank
                np.random.gamma(2, 3, 50),  # Non-toxic - lower rank
            ]
        ),
        "model": ["test-model"] * 100,
    # synthetic SMILES strings to allow length analysis
    "smiles": ["C" * int(x) for x in (np.random.randint(5, 40, 100))],
    }

    df = pd.DataFrame(sample_data)
    analyzer = ToxicityAnalyzer(df)

    # Generate report
    report = analyzer.generate_summary_report()
    print(report)

    # Create visualizations
    analyzer.create_visualizations("toxicity_analysis.pdf")

    # Write JSON summary (concise)
    analyzer.write_json_report("toxicity_summary.json")
    print("Wrote toxicity_summary.json")

    return analyzer


if __name__ == "__main__":
    analyzer = test_analyzer()
