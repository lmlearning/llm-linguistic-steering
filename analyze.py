import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import argparse  # Import argparse

def create_subject_groups():
    """Defines broad academic categories for MMLU subjects."""
    return {
        "STEM": ["abstract_algebra", "college_biology", "college_chemistry", "college_computer_science", 
                 "college_mathematics", "college_physics", "computer_security", "electrical_engineering", 
                 "elementary_mathematics", "high_school_biology", "high_school_chemistry", 
                 "high_school_computer_science", "high_school_mathematics", "high_school_physics", 
                 "high_school_statistics", "machine_learning", "virology", "anatomy", "astronomy",
                 "medical_genetics", "nutrition", "conceptual_physics"],
        "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history", 
                       "high_school_world_history", "philosophy", "prehistory", "world_religions", "moral_disputes",
                       "moral_scenarios", "logical_fallacies"],
        "Social Sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics", 
                            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology", 
                            "professional_psychology", "public_relations", "security_studies", "sociology", 
                            "us_foreign_policy", "global_facts"],
        "Professional": ["business_ethics", "clinical_knowledge", "college_medicine", "human_aging", "human_sexuality",
                         "international_law", "jurisprudence", "management", "marketing", "professional_accounting", 
                         "professional_law", "professional_medicine"]
    }

def analyze_and_visualize_complete(input_path: str, output_dir=None):
    """
    Loads experimental data, performs a full analysis (basic and advanced), 
    and generates a summary JSON, a Markdown table, and multiple plots.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at '{input_path}'")

    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = output_dir or f"{base_name}_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Outputs will be saved to '{output_dir}/'")
    
    agg_df = pd.DataFrame.from_dict(data['aggregated_results'], orient='index').reset_index().rename(columns={'index': 'adjective'})
    per_question_results = data.get('per_question_results', {})
    adjectives = list(data['adjective_mapping'].keys())

    # --- Analysis Layer 1: Subject and Group Sensitivity ---
    print("Analyzing subject sensitivity...")
    subject_groups = create_subject_groups()
    group_map = {subj: group for group, subjs in subject_groups.items() for subj in subjs}
    
    subject_sensitivities = defaultdict(list)
    for q_key, shap_values in per_question_results.items():
        subject = '_'.join(q_key.split('_')[2:])
        if subject and shap_values:
            subject_sensitivities[subject].append(np.sum(np.abs(shap_values)))

    mean_subject_sensitivity = {s: np.mean(v) if v else 0 for s, v in subject_sensitivities.items()}
    
    group_sensitivity = defaultdict(list)
    for subject, sensitivity in mean_subject_sensitivity.items():
        group = group_map.get(subject, "Other")
        group_sensitivity[group].append(sensitivity)
        
    mean_group_sensitivity = {g: np.mean(v) if v else 0 for g, v in group_sensitivity.items()}
    
    # --- Analysis Layer 2: Adjective Correlations ---
    print("Analyzing adjective correlations...")
    if per_question_results:
        shap_matrix = np.array([list(q_vals) for q_vals in per_question_results.values()]).T
        shap_df = pd.DataFrame(shap_matrix, index=adjectives)
        correlation_matrix = shap_df.T.corr()
    else:
        correlation_matrix = pd.DataFrame()

    # --- Generate Summary JSON ---
    summary_data = {
        "parameters": data["parameters"],
        "analysis_summary": {
            "top_10_positive_adjectives": agg_df.sort_values('mean_shapley', ascending=False).head(10)['adjective'].tolist(),
            "top_10_negative_adjectives": agg_df.sort_values('mean_shapley', ascending=True).head(10)['adjective'].tolist(),
            "top_10_impactful_adjectives_abs": agg_df.sort_values('mean_abs_shapley', ascending=False).head(10)['adjective'].tolist(),
            "mean_sensitivity_by_group": sorted(mean_group_sensitivity.items(), key=lambda x: x[1], reverse=True),
            "top_10_most_volatile_adjectives": agg_df.sort_values('std_shapley', ascending=False).head(10)['adjective'].tolist(),
            "top_10_most_stable_adjectives": agg_df.sort_values('std_shapley', ascending=True).head(10)['adjective'].tolist()
        }
    }
    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to {summary_path}")

    # --- Generate Summary Markdown Table ---
    print("Generating summary Markdown table...")
    table_df = agg_df.sort_values('mean_abs_shapley', ascending=False)
    table_path = os.path.join(output_dir, f"{base_name}_summary_table.md")
    with open(table_path, 'w') as f:
        f.write("# Adjective Impact Analysis Summary\n\n")
        f.write(f"Analysis of results from `{base_name}.json`.\n\n")
        f.write("This table ranks all adjectives by their overall impact (Mean Absolute Shapley Value).\n\n")
        f.write(table_df.to_markdown(index=False, floatfmt=".4f"))
    print(f"Summary table saved to {table_path}")

    # --- Generate Plots ---
    print("Generating plots...")
    sns.set_theme(style="whitegrid")

    # Plot 1: Top Positive and Negative Adjectives
    top_n = 15
    df_sorted_mean = agg_df.sort_values(by='mean_shapley', ascending=False)
    plot_df = pd.concat([df_sorted_mean.head(top_n), df_sorted_mean.tail(top_n)])
    
    palette_dict = {
        adj: '#2ca02c' if shap > 0 else '#d62728' 
        for adj, shap in zip(plot_df['adjective'], plot_df['mean_shapley'])
    }
    
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='mean_shapley', y='adjective', data=plot_df.sort_values('mean_shapley', ascending=False),
        hue='adjective', palette=palette_dict, legend=False, dodge=False
    )
    plt.title(f'Top {top_n} Adjectives that Steer Towards (+) or Away From (-) Correctness', fontsize=16, pad=20)
    plt.xlabel('Mean Shapley Value (Average Steering Direction)', fontsize=12)
    plt.ylabel('Adjective', fontsize=12)
    plt.axvline(0, color='black', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_1_mean_shapley.png"), dpi=300)
    plt.close()

    # Plot 2: Top Adjectives by Mean Absolute Shapley Value
    plt.figure(figsize=(12, 10))
    abs_plot_df = agg_df.sort_values('mean_abs_shapley', ascending=False).head(20)
    sns.barplot(x='mean_abs_shapley', y='adjective', hue='adjective', data=abs_plot_df, palette='viridis_r', legend=False)
    plt.title('Top 20 Most Influential Adjectives (Magnitude of Steering)', fontsize=16, pad=20)
    plt.xlabel('Mean Absolute Shapley Value (Overall Impact)', fontsize=12)
    plt.ylabel('Adjective', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_2_mean_abs_shapley.png"), dpi=300)
    plt.close()
    
    # Plot 3: Distribution of Overall Impact (Log Scale)
    plt.figure(figsize=(12, 8))
    sorted_abs = agg_df.sort_values('mean_abs_shapley', ascending=False)
    sns.lineplot(x=range(len(sorted_abs)), y=sorted_abs['mean_abs_shapley'], marker='o', lw=2, color='#4c72b0')
    plt.title('Distribution of Adjective Impact (The Long Tail)', fontsize=16, pad=20)
    plt.xlabel('Adjective Rank (by Impact)', fontsize=12)
    plt.ylabel('Mean Absolute Shapley Value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_3_impact_distribution.png"), dpi=300)
    plt.close()

    # Plot 4: Quadrant Analysis with Labels
    plt.figure(figsize=(13, 13))
    sns.scatterplot(data=agg_df, x='std_shapley', y='mean_shapley', s=120, hue='mean_abs_shapley', palette='viridis', legend='brief')
    median_std = agg_df['std_shapley'].median()
    plt.axhline(0, color='grey', linestyle='--', lw=1.5)
    plt.axvline(median_std, color='grey', linestyle='--', lw=1.5)
    
    annotations = ['active', 'academic', 'plain', 'bold', 'zany', 'flawless', 'clear', 'useful', 'formal', 'scientific']
    for _, row in agg_df[agg_df['adjective'].isin(annotations)].iterrows():
        plt.text(row['std_shapley'] * 1.05, row['mean_shapley'], row['adjective'], fontsize=11, weight='bold')

    plt.title('Adjective Steering Profile: Direction vs. Volatility', fontsize=18, pad=20)
    plt.xlabel('Volatility (Standard Deviation of Shapley Value)', fontsize=14)
    plt.ylabel('Average Directional Steering (Mean Shapley Value)', fontsize=14)
    
    # Add quadrant labels
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(xlim[0] + (median_std - xlim[0]) * 0.1, ylim[1] * 0.8, 'Reliable Enhancers\n(Low Volatility, Positive Impact)', 
             fontsize=12, alpha=0.8, ha='left', va='top', color='darkgreen')
    plt.text(xlim[1] - (xlim[1] - median_std) * 0.1, ylim[1] * 0.8, 'Volatile Enhancers\n(High Volatility, Positive Impact)', 
             fontsize=12, alpha=0.8, ha='right', va='top', color='darkgreen')
    plt.text(xlim[0] + (median_std - xlim[0]) * 0.1, ylim[0] * 0.8, 'Reliable Disruptors\n(Low Volatility, Negative Impact)', 
             fontsize=12, alpha=0.8, ha='left', va='bottom', color='darkred')
    plt.text(xlim[1] - (xlim[1] - median_std) * 0.1, ylim[0] * 0.8, 'Volatile Disruptors\n(High Volatility, Negative Impact)', 
             fontsize=12, alpha=0.8, ha='right', va='bottom', color='darkred')
             
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_4_quadrant_analysis.png"), dpi=300)
    plt.close()

    # Plot 5: Grouped Subject Sensitivity
    if mean_group_sensitivity:
        group_df = pd.DataFrame(sorted(mean_group_sensitivity.items(), key=lambda x: x[1], reverse=True),
                                columns=['Domain', 'Sensitivity'])
        plt.figure(figsize=(10, 7))
        sns.barplot(x='Sensitivity', y='Domain', hue='Domain', data=group_df, palette='magma', legend=False)
        plt.title('Sensitivity to Adjectival Steering by Academic Domain', fontsize=16, pad=20)
        plt.xlabel('Average Steering Magnitude (Sum of Abs. Shapley Values)', fontsize=12)
        plt.ylabel('Domain', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_5_domain_sensitivity.png"), dpi=300)
        plt.close()

    # Plot 6: Adjective Correlation Heatmap
    if not correlation_matrix.empty:
        interesting_adjectives = sorted(list(set(agg_df.nlargest(15, 'mean_abs_shapley')['adjective'].tolist() + 
                                               ['academic', 'scientific', 'simple', 'complex', 'brief', 'thorough'])))
        interesting_adjectives = [adj for adj in interesting_adjectives if adj in correlation_matrix.index]
        corr_subset = correlation_matrix.loc[interesting_adjectives, interesting_adjectives]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_subset, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
        plt.title('Correlation of Adjective Steering Effects Across Questions', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_6_correlation_heatmap.png"), dpi=300)
        plt.close()
        
    print("\nComplete analysis finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Shapley value results from an LLM experiment.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file (e.g., o3_corrected.json).")
    parser.add_argument("--output-dir", help="Directory for the generated report (default: INPUT_analysis)")
    args = parser.parse_args()
    
    try:
        analyze_and_visualize_complete(args.input_file, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
