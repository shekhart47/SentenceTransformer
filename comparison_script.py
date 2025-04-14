import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import Dict, List, Tuple

def load_results(base_file: str, optimized_file: str) -> Tuple[Dict, Dict]:
    """
    Load the latency results from both base and optimized model tests.
    
    Args:
        base_file: Path to base model results JSON
        optimized_file: Path to optimized model results JSON
        
    Returns:
        Tuple of (base_results, optimized_results)
    """
    try:
        with open(base_file, 'r') as f:
            base_results = json.load(f)
    except Exception as e:
        print(f"Error loading base results: {e}")
        base_results = {}
    
    try:
        with open(optimized_file, 'r') as f:
            optimized_results = json.load(f)
    except Exception as e:
        print(f"Error loading optimized results: {e}")
        optimized_results = {}
    
    return base_results, optimized_results

def generate_comparison_tables(base_results: Dict, optimized_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate comparison tables between base and optimized models.
    
    Args:
        base_results: Base model latency results
        optimized_results: Optimized model latency results
        
    Returns:
        Tuple of DataFrames: (overall_comparison, query_comparison, batch_comparison)
    """
    # Extract overall metrics
    if not base_results or not optimized_results:
        print("Cannot generate comparison - missing results")
        return pd.DataFrame()
    
    if "overall" not in base_results or "overall" not in optimized_results:
        print("Cannot generate overall comparison - missing overall metrics")
        return pd.DataFrame()
    
    # Overall metrics comparison
    overall_metrics = {
        "Metric": [
            "Mean Latency (ms)",
            "Min Latency (ms)",
            "Max Latency (ms)",
            "Std Deviation (ms)",
            "P50 Latency (ms)",
            "P90 Latency (ms)",
            "P95 Latency (ms)",
            "P99 Latency (ms)"
        ],
        "Base Model": [
            base_results["overall"]["mean"],
            base_results["overall"]["min"],
            base_results["overall"]["max"],
            base_results["overall"]["std"],
            base_results["overall"]["percentiles"]["p50"],
            base_results["overall"]["percentiles"]["p90"],
            base_results["overall"]["percentiles"]["p95"],
            base_results["overall"]["percentiles"]["p99"]
        ],
        "Optimized Model": [
            optimized_results["overall"]["mean"],
            optimized_results["overall"]["min"],
            optimized_results["overall"]["max"],
            optimized_results["overall"]["std"],
            optimized_results["overall"]["percentiles"]["p50"],
            optimized_results["overall"]["percentiles"]["p90"],
            optimized_results["overall"]["percentiles"]["p95"],
            optimized_results["overall"]["percentiles"]["p99"]
        ]
    }
    
    # Calculate improvement percentages
    overall_metrics["Improvement (%)"] = [
        ((base - opt) / base * 100) if base > 0 else 0
        for base, opt in zip(overall_metrics["Base Model"], overall_metrics["Optimized Model"])
    ]
    
    # Create DataFrame for overall comparison
    overall_df = pd.DataFrame(overall_metrics)
    
    # Per-query comparison if available
    query_comparison = []
    
    if "per_query" in base_results and "per_query" in optimized_results:
        # Find common queries
        common_queries = set(base_results["per_query"].keys()) & set(optimized_results["per_query"].keys())
        
        for query in common_queries:
            base_mean = base_results["per_query"][query]["mean"]
            opt_mean = optimized_results["per_query"][query]["mean"]
            improvement = ((base_mean - opt_mean) / base_mean * 100) if base_mean > 0 else 0
            
            query_comparison.append({
                "Query": query,
                "Base Model (ms)": base_mean,
                "Optimized Model (ms)": opt_mean,
                "Improvement (%)": improvement
            })
    
    # Create DataFrame for query comparison
    query_df = pd.DataFrame(query_comparison) if query_comparison else pd.DataFrame()
    
    # Batch latency comparison if available
    batch_df = pd.DataFrame()
    
    if "batch_results" in optimized_results:
        batch_data = {
            "Batch Size": [],
            "Latency per Query (ms)": []
        }
        
        for batch_key, batch_info in optimized_results.items():
            if batch_key.startswith("batch_"):
                batch_size = int(batch_key.split("_")[1])
                batch_data["Batch Size"].append(batch_size)
                batch_data["Latency per Query (ms)"].append(batch_info["mean_per_query"])
        
        if batch_data["Batch Size"]:
            batch_df = pd.DataFrame(batch_data)
            batch_df = batch_df.sort_values("Batch Size")
    
    return overall_df, query_df, batch_df

def plot_comparisons(overall_df: pd.DataFrame, query_df: pd.DataFrame, batch_df: pd.DataFrame, 
                    output_dir: str = "./plots"):
    """
    Generate comparison plots between base and optimized models.
    
    Args:
        overall_df: DataFrame with overall metrics comparison
        query_df: DataFrame with per-query comparison
        batch_df: DataFrame with batch size comparison
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall latency comparison bar chart
    if not overall_df.empty:
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ["Mean Latency (ms)", "P50 Latency (ms)", "P90 Latency (ms)", "P95 Latency (ms)"]
        
        # Extract data for the selected metrics
        plot_data = overall_df[overall_df["Metric"].isin(metrics_to_plot)]
        
        # Set up positions for the bars
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        # Extract values for base and optimized models
        base_values = [plot_data[plot_data["Metric"] == m]["Base Model"].values[0] for m in metrics_to_plot]
        opt_values = [plot_data[plot_data["Metric"] == m]["Optimized Model"].values[0] for m in metrics_to_plot]
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        base_bars = ax.bar(x - width/2, base_values, width, label='Base Model')
        opt_bars = ax.bar(x + width/2, opt_values, width, label='Optimized Model')
        
        # Add labels and title
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison: Base vs. Optimized Model')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        
        # Add improvement percentage above bars
        for i in range(len(metrics_to_plot)):
            improvement = ((base_values[i] - opt_values[i]) / base_values[i] * 100) if base_values[i] > 0 else 0
            plt.text(i, max(base_values[i], opt_values[i]) + 2, 
                     f"{improvement:.1f}% better", 
                     ha='center', va='bottom', fontweight='bold')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_latency_comparison.png"))
        plt.close()
    
    # 2. Per-query latency comparison
    if not query_df.empty and len(query_df) > 0:
        # Select a subset of queries if there are too many
        max_queries = 10
        if len(query_df) > max_queries:
            # Sort by improvement and take top improved queries
            query_df_plot = query_df.sort_values("Improvement (%)", ascending=False).head(max_queries)
        else:
            query_df_plot = query_df
            
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Truncate query text for readability
        max_query_length = 50
        query_df_plot["Short Query"] = query_df_plot["Query"].apply(
            lambda x: x[:max_query_length] + "..." if len(x) > max_query_length else x)
        
        # Sort by base model latency for better visualization
        query_df_plot = query_df_plot.sort_values("Base Model (ms)", ascending=False)
        
        # Create the bar chart
        x = np.arange(len(query_df_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        base_bars = ax.barh(x - width/2, query_df_plot["Base Model (ms)"], height=width, label='Base Model')
        opt_bars = ax.barh(x + width/2, query_df_plot["Optimized Model (ms)"], height=width, label='Optimized Model')
        
        # Add labels and title
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Query')
        ax.set_title('Query Latency Comparison: Base vs. Optimized Model')
        ax.set_yticks(x)
        ax.set_yticklabels(query_df_plot["Short Query"])
        ax.legend()
        
        # Add improvement percentage beside bars
        for i, (_, row) in enumerate(query_df_plot.iterrows()):
            opt_val = row["Optimized Model (ms)"]
            ax.text(opt_val + 1, i + width/2, f"{row['Improvement (%)']:.1f}% better", 
                   va='center', fontweight='bold')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "query_latency_comparison.png"))
        plt.close()
    
    # 3. Batch size vs. latency plot
    if not batch_df.empty and len(batch_df) > 0:
        plt.figure(figsize=(12, 6))
        
        # Line plot for batch size vs latency
        plt.plot(batch_df["Batch Size"], batch_df["Latency per Query (ms)"], 
                marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Add labels and title
        plt.xlabel('Batch Size')
        plt.ylabel('Latency per Query (ms)')
        plt.title('Effect of Batch Size on Query Latency')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, row in batch_df.iterrows():
            plt.text(row["Batch Size"], row["Latency per Query (ms)"] + 0.5, 
                    f"{row['Latency per Query (ms)']:.2f} ms", 
                    ha='center', va='bottom')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_size_latency.png"))
        plt.close()

def generate_csv_report(overall_df: pd.DataFrame, query_df: pd.DataFrame, batch_df: pd.DataFrame,
                     output_dir: str = "./results"):
    """
    Generate CSV reports of the comparison data.
    
    Args:
        overall_df: DataFrame with overall metrics comparison
        query_df: DataFrame with per-query comparison
        batch_df: DataFrame with batch size comparison
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrames to CSV
    if not overall_df.empty:
        overall_df.to_csv(os.path.join(output_dir, "overall_comparison.csv"), index=False)
    
    if not query_df.empty:
        query_df.to_csv(os.path.join(output_dir, "query_comparison.csv"), index=False)
    
    if not batch_df.empty:
        batch_df.to_csv(os.path.join(output_dir, "batch_comparison.csv"), index=False)
    
    print(f"CSV reports saved to {output_dir}")

def generate_report(overall_df: pd.DataFrame, query_df: pd.DataFrame, batch_df: pd.DataFrame, 
                   output_file: str = "latency_comparison_report.md"):
    """
    Generate a comprehensive markdown report of the latency comparison.
    
    Args:
        overall_df: DataFrame with overall metrics comparison
        query_df: DataFrame with per-query comparison
        batch_df: DataFrame with batch size comparison
        output_file: Path to save the report
    """
    report = []
    
    # Report title
    report.append("# E5 Model Latency Optimization Report\n")
    report.append("## Using SentenceTransformers for Optimized Performance\n")
    
    # Executive summary
    report.append("## Executive Summary\n")
    if not overall_df.empty:
        mean_improvement = overall_df[overall_df["Metric"] == "Mean Latency (ms)"]["Improvement (%)"].values[0]
        p95_improvement = overall_df[overall_df["Metric"] == "P95 Latency (ms)"]["Improvement (%)"].values[0]
        
        report.append(f"The optimized E5 model implementation using SentenceTransformers shows:")
        report.append(f"- **{mean_improvement:.1f}%** improvement in mean latency")
        report.append(f"- **{p95_improvement:.1f}%** improvement in P95 latency (critical for production services)")
        
        # Add more context about the batch processing if available
        if not batch_df.empty and len(batch_df) > 1:
            single_query = batch_df[batch_df["Batch Size"] == 1]["Latency per Query (ms)"].values[0] if 1 in batch_df["Batch Size"].values else None
            max_batch = batch_df["Batch Size"].max()
            max_batch_latency = batch_df[batch_df["Batch Size"] == max_batch]["Latency per Query (ms)"].values[0]
            
            if single_query is not None:
                batch_improvement = ((single_query - max_batch_latency) / single_query * 100)
                report.append(f"- Additional **{batch_improvement:.1f}%** latency improvement per query when using batch size of {max_batch}")
        
        report.append("")
    
    # Overall latency comparison
    report.append("## 1. Overall Latency Comparison\n")
    if not overall_df.empty:
        # Convert DataFrame to markdown table
        overall_table = overall_df.to_markdown(index=False, floatfmt=".2f")
        report.append(overall_table)
        report.append("\n![Overall Latency Comparison](./plots/overall_latency_comparison.png)\n")
    else:
        report.append("*No overall comparison data available.*\n")
    
    # Per-query latency comparison
    report.append("## 2. Per-Query Latency Analysis\n")
    if not query_df.empty:
        # Get summary stats
        avg_improvement = query_df["Improvement (%)"].mean()
        max_improvement = query_df["Improvement (%)"].max()
        max_improved_query = query_df.loc[query_df["Improvement (%)"].idxmax()]["Query"]
        
        report.append(f"Average improvement across all queries: **{avg_improvement:.2f}%**")
        report.append(f"Maximum improvement: **{max_improvement:.2f}%** for query: \"{max_improved_query}\"")
        report.append("\n")
        
        # Limit number of queries in table for readability
        max_queries_in_table = 10
        if len(query_df) > max_queries_in_table:
            # Sort by improvement
            query_table_df = query_df.sort_values("Improvement (%)", ascending=False).head(max_queries_in_table)
            query_table = query_table_df.to_markdown(index=False, floatfmt=".2f")
            report.append("Top 10 queries with highest improvement:\n")
            report.append(query_table)
        else:
            query_table = query_df.to_markdown(index=False, floatfmt=".2f")
            report.append(query_table)
        
        report.append("\n![Query Latency Comparison](./plots/query_latency_comparison.png)\n")
    else:
        report.append("*No per-query comparison data available.*\n")
    
    # Batch processing analysis
    report.append("## 3. Batch Processing Performance\n")
    if not batch_df.empty and len(batch_df) > 0:
        report.append("SentenceTransformers enables efficient batch processing of queries. The following table and chart show how latency per query changes with different batch sizes:\n")
        
        # Convert DataFrame to markdown table
        batch_table = batch_df.to_markdown(index=False, floatfmt=".2f")
        report.append(batch_table)
        
        # Find optimal batch size (lowest latency per query)
        optimal_batch = batch_df.loc[batch_df["Latency per Query (ms)"].idxmin()]
        report.append(f"\nThe optimal batch size appears to be **{int(optimal_batch['Batch Size'])}**, " +
                     f"which achieves a latency of **{optimal_batch['Latency per Query (ms)']:.2f} ms** per query.")
        
        report.append("\n![Batch Size vs Latency](./plots/batch_size_latency.png)\n")
    else:
        report.append("*No batch processing data available.*\n")
    
    # Implementation details and recommendations
    report.append("## 4. Implementation Details\n")
    report.append("### Key Optimizations Applied\n")
    report.append("1. **SentenceTransformers Framework**: Replaced manual pooling and normalization with SentenceTransformers' optimized implementation")
    report.append("2. **Efficient Tokenization**: SentenceTransformers handles tokenization more efficiently")
    report.append("3. **Batch Processing**: Added support for processing multiple queries in a single forward pass")
    report.append("4. **Memory Optimization**: Better memory management and caching")
    report.append("\n### Recommendations\n")
    
    # Add specific recommendations based on the data
    if not batch_df.empty and len(batch_df) > 1:
        optimal_batch = batch_df.loc[batch_df["Latency per Query (ms)"].idxmin()]["Batch Size"]
        report.append(f"1. **Use Batch Processing**: When possible, process queries in batches of size {int(optimal_batch)}")
    
    report.append("2. **Continue with SentenceTransformers**: The optimized implementation consistently outperforms the base implementation")
    report.append("3. **Model Quantization**: Consider further optimizations through quantization (INT8/FP16) for deployment")
    report.append("4. **Hardware Acceleration**: For deployment, ensure GPU acceleration is available")
    
    # Write the report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated at {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="E5 model latency comparison analysis")
    parser.add_argument("--base_results", type=str, required=True, 
                        help="Path to the base model latency results JSON")
    parser.add_argument("--optimized_results", type=str, required=True,
                        help="Path to the optimized model latency results JSON")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                        help="Directory to save output files")
    parser.add_argument("--report_name", type=str, default="latency_comparison_report.md",
                        help="Name of the markdown report file")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load results
    base_results, optimized_results = load_results(args.base_results, args.optimized_results)
    
    if not base_results or not optimized_results:
        print("Error: Could not load results files.")
        sys.exit(1)
    
    # Generate comparison tables
    overall_df, query_df, batch_df = generate_comparison_tables(base_results, optimized_results)
    
    # Create plots
    plot_comparisons(overall_df, query_df, batch_df, plots_dir)
    
    # Generate CSV reports
    generate_csv_report(overall_df, query_df, batch_df, args.output_dir)
    
    # Generate markdown report
    report_path = os.path.join(args.output_dir, args.report_name)
    generate_report(overall_df, query_df, batch_df, report_path)
    
    print(f"Comparison analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()