import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
import tkinter as tk
from tkinter import filedialog, messagebox
from statsmodels.stats.anova import AnovaRM


def select_directory():
    """
    Open directory selection dialog.

    Args:
        title: Dialog title

    Returns:
        Selected directory path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()

    # Use default directory from settings if available
    try:
        directory_path = filedialog.askopenfilename(
            title="Select Results File to Analyze",
            initialdir=os.getcwd()
        )

        if directory_path:
            return directory_path
        else:
            return None

    except Exception as e:
        return None
    finally:
        root.destroy()

# Load and clean the data
def load_and_clean_data(filepath):
    """Load and preprocess the histology data"""
    df = pd.read_csv(filepath)
    
    # Remove rows with missing key data
    required_cols = ['group', 'scar', 'annotation_area', 'total_connexin_area', 
                    'total_connexin_area_per_cell', 'total_mean_plaque_size', 'total_nuclei',
                     'total_connexins_per_annotation_area']
    df_clean = df.dropna(subset=required_cols)
    
    # Standardize scar categories (handle case inconsistencies)
    df_clean['scar'] = df_clean['scar'].str.lower()
    
    # Calculate derived metrics
    df_clean['connexin_area_per_annotation'] = df_clean['total_connexin_area'] / df_clean['annotation_area']
    df_clean['connexin_count_per_annotation'] = df_clean['total_connexins_per_annotation_area']
    df_clean['connexin_area_per_cell'] = df_clean['total_connexin_area_per_cell']
    df_clean['mean_plaque_size'] = df_clean['total_mean_plaque_size']
    df_clean['cell_density'] = df_clean['total_nuclei'] / df_clean['annotation_area'] # Should be correlated with fibrosis
    #df_clean['lateralization_by_area'] =
    #df_clean['lateralization_by_count'] =
    
    # Create group_scar combinations for analysis
    df_clean['group_scar'] = df_clean['group'] + '_' + df_clean['scar']
    
    return df_clean

def get_comparison_groups(df):
    """Dynamically determine comparison groups from the data"""
    # Get unique group-scar combinations
    group_scar_combinations = df['group_scar'].unique()
    
    # Define comparison categories
    comparisons = {
        'Healthy_Tissue': [combo for combo in group_scar_combinations if '_healthy' in combo],
        'Border_Tissue': [combo for combo in group_scar_combinations if '_border' in combo],
        'Center_Scar': [combo for combo in group_scar_combinations if '_center' in combo]
    }
    
    # Filter out empty categories
    comparisons = {k: v for k, v in comparisons.items() if len(v) > 0}
    
    return comparisons

def calculate_summary_statistics(df, comparisons, metrics):
    """Calculate summary statistics for all groups and metrics"""
    results = []
    
    for comparison_name, groups in comparisons.items():
        for group in groups:
            group_data = df[df['group_scar'] == group]
            
            for metric in metrics:
                data = group_data[metric].dropna()
                
                if len(data) > 0:
                    results.append({
                        'Comparison_Category': comparison_name,
                        'Group': group,
                        'Metric': metric,
                        'N': len(data),
                        'Mean': np.mean(data),
                        'Std': np.std(data, ddof=1) if len(data) > 1 else 0,
                        'Median': np.median(data),
                        'Min': np.min(data),
                        'Max': np.max(data),
                        'SE': np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0
                    })
    
    return pd.DataFrame(results)

def perform_anova_analysis(df, healthy_groups, metrics):
    """Perform one-way ANOVA for healthy tissue comparisons"""
    anova_results = []
    
    for metric in metrics:
        # Get data for each group
        group_data = []
        group_info = []
        
        for group in healthy_groups:
            data = df[df['group_scar'] == group][metric].dropna()
            if len(data) > 0:
                group_data.append(data)
                group_info.append({
                    'group': group,
                    'n': len(data),
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1) if len(data) > 1 else 0
                })
        
        # Perform ANOVA if we have at least 2 groups with data
        if len(group_data) >= 2:
            f_stat, p_value = f_oneway(*group_data)
            
            # Calculate degrees of freedom
            df_between = len(group_data) - 1
            df_within = sum(len(group) for group in group_data) - len(group_data)
            
            anova_results.append({
                'Metric': metric,
                'F_statistic': f_stat,
                'p_value': p_value,
                'df_between': df_between,
                'df_within': df_within,
                'groups': group_info
            })
    
    return anova_results

def perform_ttest_analysis(df, comparison_pairs, metrics):
    """Perform t-tests for pairwise comparisons"""
    ttest_results = []
    
    for comparison_name, (group1, group2) in comparison_pairs.items():
        for metric in metrics:
            data1 = df[df['group_scar'] == group1][metric].dropna()
            data2 = df[df['group_scar'] == group2][metric].dropna()
            
            if len(data1) > 0 and len(data2) > 0:
                # Perform Welch's t-test (unequal variances)
                t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                    (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                   (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                
                ttest_results.append({
                    'Comparison': comparison_name,
                    'Metric': metric,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'group1': group1,
                    'group1_mean': np.mean(data1),
                    'group1_std': np.std(data1, ddof=1) if len(data1) > 1 else 0,
                    'group1_n': len(data1),
                    'group2': group2,
                    'group2_mean': np.mean(data2),
                    'group2_std': np.std(data2, ddof=1) if len(data2) > 1 else 0,
                    'group2_n': len(data2),
                    'mean_difference': np.mean(data1) - np.mean(data2),
                    'cohens_d': cohens_d
                })
    
    return pd.DataFrame(ttest_results)

def analyze_correlations(df, correlation_pairs):
    """Analyze correlations between variables"""
    correlation_results = []
    
    for var1, var2 in correlation_pairs:
        # Remove rows where either variable is missing
        clean_data = df[[var1, var2]].dropna()
        
        if len(clean_data) > 2:
            r, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
            
            correlation_results.append({
                'Variable_1': var1,
                'Variable_2': var2,
                'Correlation_r': r,
                'p_value': p_value,
                'N': len(clean_data)
            })
    
    return pd.DataFrame(correlation_results)

def print_sample_sizes(df):
    """Print sample sizes for each group"""
    print("SAMPLE SIZES BY GROUP:")
    print("=" * 50)
    
    group_counts = df['group_scar'].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"{group}: {count} samples")
    
    print(f"\nTotal samples analyzed: {len(df)}")
    return group_counts

def print_anova_results(anova_results):
    """Print ANOVA results in a formatted way"""
    print("\n\nONE-WAY ANOVA RESULTS (Healthy Tissue Comparison):")
    print("=" * 70)
    
    for result in anova_results:
        print(f"\n{result['Metric'].upper().replace('_', ' ')}:")
        print(f"F({result['df_between']}, {result['df_within']}) = {result['F_statistic']:.4f}, p = {result['p_value']:.4f}")
        
        for group_info in result['groups']:
            print(f"  {group_info['group']}: {group_info['mean']:.6f} ± {group_info['std']:.6f} (n={group_info['n']})")
        
        # Interpret significance
        if result['p_value'] < 0.001:
            print("  *** HIGHLY SIGNIFICANT (p < 0.001)")
        elif result['p_value'] < 0.01:
            print("  ** SIGNIFICANT (p < 0.01)")
        elif result['p_value'] < 0.05:
            print("  * SIGNIFICANT (p < 0.05)")
        else:
            print("  Not significant (p ≥ 0.05)")

def print_ttest_results(ttest_df):
    """Print t-test results in a formatted way"""
    print("\n\nT-TEST RESULTS:")
    print("=" * 50)
    
    for comparison in ttest_df['Comparison'].unique():
        print(f"\n{comparison.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        comp_data = ttest_df[ttest_df['Comparison'] == comparison]
        
        for _, row in comp_data.iterrows():
            print(f"\n{row['Metric'].upper().replace('_', ' ')}:")
            print(f"t = {row['t_statistic']:.4f}, p = {row['p_value']:.4f}")
            print(f"  {row['group1']}: {row['group1_mean']:.6f} ± {row['group1_std']:.6f} (n={row['group1_n']})")
            print(f"  {row['group2']}: {row['group2_mean']:.6f} ± {row['group2_std']:.6f} (n={row['group2_n']})")
            print(f"  Mean difference: {row['mean_difference']:.6f}")
            print(f"  Effect size (Cohen's d): {row['cohens_d']:.3f}")
            
            # Interpret significance
            if row['p_value'] < 0.001:
                print("  *** HIGHLY SIGNIFICANT (p < 0.001)")
            elif row['p_value'] < 0.01:
                print("  ** SIGNIFICANT (p < 0.01)")
            elif row['p_value'] < 0.05:
                print("  * SIGNIFICANT (p < 0.05)")
            else:
                print("  Not significant (p ≥ 0.05)")

def identify_key_findings(ttest_df, anova_results):
    """Automatically identify and summarize key findings"""
    print("\n\nKEY FINDINGS SUMMARY:")
    print("=" * 60)
    
    # Find significant results
    significant_ttests = ttest_df[ttest_df['p_value'] < 0.05]
    significant_anovas = [r for r in anova_results if r['p_value'] < 0.05]
    
    print(f"\n1. SIGNIFICANT FINDINGS ({len(significant_ttests)} significant comparisons found):")
    
    if len(significant_ttests) > 0:
        for _, row in significant_ttests.iterrows():
            direction = "higher" if row['mean_difference'] > 0 else "lower"
            print(f"   • {row['Comparison']}: {row['group1']} shows significantly {direction}")
            print(f"     {row['Metric'].replace('_', ' ')} than {row['group2']} (p = {row['p_value']:.4f})")
    else:
        print("   • No significant differences found in pairwise comparisons")
    
    if len(significant_anovas) > 0:
        print(f"\n2. SIGNIFICANT ANOVA RESULTS:")
        for result in significant_anovas:
            print(f"   • {result['Metric'].replace('_', ' ')}: F = {result['F_statistic']:.3f}, p = {result['p_value']:.4f}")
    
    # Identify largest effect sizes
    print(f"\n3. LARGEST EFFECT SIZES:")
    ttest_df['abs_cohens_d'] = np.abs(ttest_df['cohens_d'])
    largest_effects = ttest_df.nlargest(3, 'abs_cohens_d')
    
    for _, row in largest_effects.iterrows():
        effect_size_interpretation = "large" if abs(row['cohens_d']) > 0.8 else "medium" if abs(row['cohens_d']) > 0.5 else "small"
        print(f"   • {row['Comparison']} - {row['Metric'].replace('_', ' ')}: Cohen's d = {row['cohens_d']:.3f} ({effect_size_interpretation})")

def create_visualizations(df, save_plots=True):
    """Create comprehensive visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Connexin Histology Analysis Results', fontsize=16, fontweight='bold')
    
    metrics = ['connexin_area_per_annotation', 'connexin_area_per_cell', 'mean_plaque_size',
               'connexin_count_per_annotation']
    metric_titles = ['Connexin Area per Annotation', 'Connexin Area per Cell', 'Mean Plaque Size per Annotation',
                     'Connexin Count per Annotation']
    
    # Top row: Box plots for the three main metrics
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        sns.boxplot(data=df, x='scar', y=metric, hue='group', ax=axes[0, i])
        axes[0, i].set_title(title, fontweight='bold')
        axes[0, i].set_xlabel('Scar Type')
        axes[0, i].tick_params(axis='x', rotation=45)
        
        # Add sample sizes to the plot
        for j, scar_type in enumerate(df['scar'].unique()):
            for k, group in enumerate(df['group'].unique()):
                n = len(df[(df['scar'] == scar_type) & (df['group'] == group)])
                if n > 0:
                    axes[0, i].text(j + (k-1)*0.3, axes[0, i].get_ylim()[1]*0.9, f'n={n}', 
                                  ha='center', va='bottom', fontsize=8)
    
    # Bottom row: Additional analyses
    
    # Plot 4: Fibrosis vs connexin correlation
    if 'Fibrosis Percentage' in df.columns:
        sns.scatterplot(data=df, x='Fibrosis Percentage', y='connexin_area_per_annotation', 
                       hue='group', style='scar', ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Fibrosis vs Connexin Area', fontweight='bold')
        axes[1, 0].set_xlabel('Fibrosis Percentage (%)')
    
    # Plot 5: Cell density comparison
    sns.boxplot(data=df, x='scar', y='cell_density', hue='group', ax=axes[1, 1])
    axes[1, 1].set_title('Cell Density by Group and Scar Type', fontweight='bold')
    axes[1, 1].set_ylabel('Cells per Annotation Area')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Location effects (if location data available)
    if 'location' in df.columns:
        sns.boxplot(data=df, x='location', y='connexin_area_per_annotation', hue='group', ax=axes[1, 2])
        axes[1, 2].set_title('Connexin by Heart Location', fontweight='bold')
        axes[1, 2].set_ylabel('Connexin Area per Annotation')
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        # Alternative plot if no location data
        sns.violinplot(data=df, x='group', y='connexin_area_per_cell', ax=axes[1, 2])
        axes[1, 2].set_title('Connexin Area per Cell Distribution', fontweight='bold')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('connexin_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        print(f"\nPlots saved as 'connexin_analysis_comprehensive.png'")
    
    plt.show()
    
    return fig

def main_analysis(filepath):
    """Main analysis function"""
    
    print("=== CONNEXIN HISTOLOGY DATA ANALYSIS ===")
    print("Loading and processing data...")
    
    # Load and clean data
    df = load_and_clean_data(filepath)
    
    # Print basic info about the dataset
    print(f"\nDataset loaded successfully!")
    print(f"Total samples after cleaning: {len(df)}")
    print(f"Groups found: {', '.join(df['group'].unique())}")
    print(f"Scar types found: {', '.join(df['scar'].unique())}")
    
    try:
        if 'location' in df.columns:
            print(f"Heart locations found: {', '.join(df['location'].unique())}")
    except Exception as e:
        print(str(e))
    
    # Get comparison groups dynamically
    comparisons = get_comparison_groups(df)
    
    # Define metrics to analyze
    metrics = ['connexin_area_per_annotation', 'connexin_area_per_cell', 'connexin_count_per_annotation',
               'mean_plaque_size']
    
    # Print sample sizes
    sample_sizes = print_sample_sizes(df)
    
    # Calculate summary statistics
    print("\n\nCalculating summary statistics...")
    summary_stats = calculate_summary_statistics(df, comparisons, metrics)
    
    # Display summary statistics table
    print("\n\nSUMMARY STATISTICS TABLE:")
    print("=" * 80)
    
    # Pivot table for better readability
    for metric in metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        metric_data = summary_stats[summary_stats['Metric'] == metric]
        
        for _, row in metric_data.iterrows():
            print(f"  {row['Group']}: {row['Mean']:.6f} ± {row['Std']:.6f} (n={row['N']})")
    
    # Perform ANOVA for healthy tissue comparison
    healthy_groups = [group for group in df['group_scar'].unique() if '_healthy' in group]
    
    if len(healthy_groups) >= 2:
        print("\n\nPerforming ANOVA analysis for healthy tissue comparison...")
        anova_results = perform_anova_analysis(df, healthy_groups, metrics)
        print_anova_results(anova_results)
    else:
        print("\n\nInsufficient healthy tissue groups for ANOVA analysis")
        anova_results = []
    
    # Perform t-tests for pairwise comparisons
    print("\n\nPerforming t-test analyses...")
    
    # Define comparison pairs dynamically
    comparison_pairs = {}
    
    # Border tissue comparison
    border_groups = [group for group in df['group_scar'].unique() if '_border' in group]
    if len(border_groups) == 2:
        comparison_pairs['Border_Tissue'] = (border_groups[0], border_groups[1])
    
    # Center scar comparison  
    center_groups = [group for group in df['group_scar'].unique() if '_center' in group]
    if len(center_groups) == 2:
        comparison_pairs['Center_Scar'] = (center_groups[0], center_groups[1])
    
    if comparison_pairs:
        ttest_results = perform_ttest_analysis(df, comparison_pairs, metrics)
        print_ttest_results(ttest_results)
        
        # Identify key findings
        identify_key_findings(ttest_results, anova_results)
    else:
        print("No suitable pairs found for t-test comparisons")
        ttest_results = pd.DataFrame()
    
    # Correlation analysis
    print("\n\nPerforming correlation analysis...")
    correlation_pairs = [
        ('Fibrosis Percentage', 'connexin_area_per_annotation'),
        ('Fibrosis Percentage', 'connexin_area_per_cell'),
        ('Fibrosis Percentage', 'cell_density'),
        ('cell_density', 'connexin_area_per_cell')
    ]
    
    # Only include correlations for variables that exist in the dataset
    available_correlations = [(var1, var2) for var1, var2 in correlation_pairs 
                            if var1 in df.columns and var2 in df.columns]
    
    if available_correlations:
        correlation_results = analyze_correlations(df, available_correlations)
        
        print("\nCORRELATION RESULTS:")
        print("-" * 40)
        for _, row in correlation_results.iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"{row['Variable_1']} vs {row['Variable_2']}: r = {row['Correlation_r']:.4f}, p = {row['p_value']:.4f} {significance} (n={row['N']})")
    
    # Create visualizations
    print("\n\nCreating visualizations...")
    create_visualizations(df)
    
    # Generate recommendations
    print("\n\nRECOMMENDATIONS FOR FURTHER ANALYSIS:")
    print("=" * 60)
    print("• Consider power analysis for sample size planning")
    print("• Examine connexin subtype distribution if data available")
    print("• Analyze spatial distribution patterns within sections")
    print("• Consider multivariate analysis with relevant covariates")
    print("• Validate findings with functional measures if possible")
    
    # Return results for further use
    return {
        'dataframe': df,
        'summary_statistics': summary_stats,
        'anova_results': anova_results,
        'ttest_results': ttest_results,
        'correlation_results': correlation_results if available_correlations else pd.DataFrame(),
        'sample_sizes': sample_sizes
    }

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    filepath = select_directory()
    
    try:
        results = main_analysis(filepath)
        print("\n\nAnalysis completed successfully!")
        print("\nResults dictionary contains:")
        print("- 'dataframe': Cleaned and processed data")
        print("- 'summary_statistics': Descriptive statistics table")
        print("- 'anova_results': ANOVA analysis results")
        print("- 'ttest_results': T-test analysis results")
        print("- 'correlation_results': Correlation analysis results")
        print("- 'sample_sizes': Sample size information")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
        print("Please update the filepath variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check your data format and try again.")
