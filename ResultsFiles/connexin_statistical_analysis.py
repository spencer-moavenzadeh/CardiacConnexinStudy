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
from scipy.stats import linregress
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')


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
    required_cols = ['group', 'scar', 'annotation_area', 'total_connexin_area', 'total_connexin_area_per_annotation_area',
                     'total_connexin_area_per_cell', 'total_mean_plaque_size', 'total_nuclei',
                     'total_connexin_count_per_annotation_area']
    df_clean = df.dropna(subset=required_cols)
    
    # Standardize scar categories (handle case inconsistencies)
    df_clean['scar'] = df_clean['scar'].str.lower()
    
    # Calculate derived metrics
    df_clean['connexin_area_per_annotation'] = df_clean['total_connexin_area_per_annotation_area']
    df_clean['connexin_count_per_annotation'] = df_clean['total_connexin_count_per_annotation_area']
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


def create_fibrosis_bins(df, bin_width_percent=10):
    """
    Create fibrosis percentage bins based on specified width

    Args:
        df: DataFrame with 'Fibrosis Percentage' column
        bin_width_percent: Width of each bin in percentage points

    Returns:
        DataFrame with added 'fibrosis_bin' column
    """
    df = df.copy()

    # Get min and max fibrosis percentages
    min_fib = df['Fibrosis Percentage'].min()
    max_fib = df['Fibrosis Percentage'].max()

    # Create bin edges
    bin_edges = np.arange(0, max_fib + bin_width_percent, bin_width_percent)
    if bin_edges[-1] < max_fib:
        bin_edges = np.append(bin_edges, max_fib + 0.01)

    # Create bins
    df['fibrosis_bin'] = pd.cut(df['Fibrosis Percentage'],
                                bins=bin_edges,
                                labels=[f"{int(i)}-{int(i + bin_width_percent)}%" for i in bin_edges[:-1]],
                                include_lowest=True)

    return df, bin_edges


def perform_stratified_two_way_anova(df, metrics, bin_width_percent=10, comparison_groups=None, exclude_groups=None):
    """
    Perform two-way ANOVA (Group × Fibrosis Bin) for specified group comparison

    Args:
        df: DataFrame with connexin data
        metrics: List of metrics to analyze
        bin_width_percent: Width of fibrosis bins in percentage points
        comparison_groups: List of groups to compare (e.g., ['infarct', 'radiation'] or ['chronicallyoccluded', 'infarct'])
                          If None, will auto-detect non-excluded groups
        exclude_groups: List of groups to exclude (e.g., ['healthy', 'control'])

    Returns:
        Dictionary with ANOVA results and post-hoc analyses
    """
    print(f"\nPerforming Stratified Two Way ANOVA with Fibrosis Bins of {bin_width_percent}%")
    print('-' * 40)

    # Filter out excluded groups
    df_filtered = df[~df['group'].isin(exclude_groups)].copy()

    # Determine comparison groups
    #if comparison_groups is None:
    #    # Auto-detect available groups (excluding specified groups)
    #    available_groups = df_filtered['group'].unique()
    #    comparison_groups = list(available_groups)
    #    print(f"Auto-detected comparison groups: {comparison_groups}")
    #else:
    #    # Use specified groups
    #    df_filtered = df_filtered[df_filtered['group'].isin(comparison_groups)].copy()

    if len(comparison_groups) < 2:
        print(f"Error: Need at least 2 groups for comparison. Found: {comparison_groups}")
        return {}

    if len(df_filtered) == 0:
        print("No data found for specified groups!")
        return {}

    #print(f"Excluded groups: {exclude_groups}")
    #print(f"Groups being compared: {comparison_groups}")

    # Create fibrosis bins
    df_binned, bin_edges = create_fibrosis_bins(df_filtered, bin_width_percent)

    # Remove rows with missing fibrosis data
    df_binned = df_binned.dropna(subset=['Fibrosis Percentage', 'fibrosis_bin'])

    results = {
        'bin_edges': bin_edges,
        'bin_width': bin_width_percent,
        'comparison_groups': comparison_groups,
        'excluded_groups': exclude_groups,
        'anova_results': [],
        'post_hoc_results': [],
        'sample_sizes': {},
        'descriptive_stats': []
    }

    # Calculate sample sizes for each group-bin combination
    sample_sizes = df_binned.groupby(['group', 'fibrosis_bin']).size().unstack(fill_value=0)
    results['sample_sizes'] = sample_sizes

    print(f"\nSample sizes by group and fibrosis bin ({bin_width_percent}% bins):")
    print(sample_sizes)

    # Check which bins have data from multiple groups for meaningful comparisons
    valid_bins = []
    for bin_name in sample_sizes.columns:
        groups_with_data = 0
        for group in comparison_groups:
            if group in sample_sizes.index:
                group_n = sample_sizes.loc[group, bin_name]
                if group_n >= 2:  # Minimum 2 samples per group
                    groups_with_data += 1

        if groups_with_data >= 2:  # At least 2 groups have sufficient data
            valid_bins.append(bin_name)

    print(f"Valid bins for comparison (≥2 samples per group, ≥2 groups): {valid_bins}")

    for metric in metrics:
        print(f"\nMETRIC: {metric.replace('_', ' ').title()}")
        print(f"Two-Way ANOVA: Group ({' vs '.join(comparison_groups)}) × Fibrosis Bin {bin_width_percent}%")
        print(f"{'.' * 40}")

        # Prepare data for analysis - only include valid bins
        analysis_data = df_binned[df_binned['fibrosis_bin'].isin(valid_bins)][
            [metric, 'group', 'fibrosis_bin']].dropna(subset=['fibrosis_bin'])
        # Drop unused categories
        analysis_data['fibrosis_bin'] = analysis_data['fibrosis_bin'].cat.remove_unused_categories()

        if len(analysis_data) < 8:  # Minimum sample size for meaningful ANOVA
            print(f"Insufficient data for {metric} (n={len(analysis_data)})")
            continue

        # Check if we have at least 2 bins and 2 groups
        n_groups = analysis_data['group'].nunique()
        n_bins = analysis_data['fibrosis_bin'].nunique()
        print(f"{n_groups} groups and {n_bins} bins of {metric}")
        print(analysis_data['fibrosis_bin'].unique())

        if n_groups < 2 or n_bins < 2:
            print(f"Insufficient factor levels (groups: {n_groups}, bins: {n_bins})")
            continue

        # Calculate descriptive statistics
        desc_stats = analysis_data[analysis_data['fibrosis_bin'].isin(valid_bins)].groupby(['group', 'fibrosis_bin'])[metric].agg(
            ['count', 'mean', 'std', 'sem']).dropna(subset=['count']).sort_values(by='fibrosis_bin').reset_index()
        results['descriptive_stats'].append({
            'metric': metric,
            'stats': desc_stats
        })

        print(f"\nDescriptive Statistics:")
        print(desc_stats.to_string(index=False))

        try:
            # Clean column names for formula - replace any problematic characters
            #clean_metric_name = metric.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')

            # Create a copy of the data with clean column names for the formula
            #formula_data = analysis_data.copy()
            #formula_data[clean_metric_name] = formula_data[metric]

            # Perform two-way ANOVA using OLS with clean column name
            #formula = f'{clean_metric_name} ~ C(group) + C(fibrosis_bin) + C(group):C(fibrosis_bin)'
            #model = ols(formula, data=formula_data).fit()
            #print(analysis_data.to_string())
            formula = f'{metric} ~ C(group) + C(fibrosis_bin) + C(group):C(fibrosis_bin)'
            model = ols(formula, data=analysis_data).fit()
            anova_table = anova_lm(model, typ=2)  # Type II ANOVA

            # Store ANOVA results
            anova_result = {
                'metric': metric,
                'anova_table': anova_table,
                'model': model,
                'formula': formula,
                'n_total': len(analysis_data),
                'n_groups': n_groups,
                'n_bins': n_bins,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'comparison_groups': comparison_groups
            }
            results['anova_results'].append(anova_result)

            print(f"\nTwo-way ANOVA Results:")
            print(anova_table)

            # Extract significance values
            group_p = anova_table.loc['C(group)', 'PR(>F)']
            bin_p = anova_table.loc['C(fibrosis_bin)', 'PR(>F)']
            interaction_p = anova_table.loc['C(group):C(fibrosis_bin)', 'PR(>F)']

            print(f"\nEffect Significance:")
            print(
                f"Main Effect - Group ({' vs '.join(comparison_groups)}): F = {anova_table.loc['C(group)', 'F']:.3f}, p = {group_p:.4f}")
            print(f"Main Effect - Fibrosis Bin: F = {anova_table.loc['C(fibrosis_bin)', 'F']:.3f}, p = {bin_p:.4f}")
            print(
                f"Interaction - Group × Fibrosis Bin: F = {anova_table.loc['C(group):C(fibrosis_bin)', 'F']:.3f}, p = {interaction_p:.4f}")

            # Interpret significance with clinical context
            print(f"\nClinical Interpretation:")
            if group_p < 0.05:
                sig_level = "***" if group_p < 0.001 else "**" if group_p < 0.01 else "*"
                print(f"{sig_level} SIGNIFICANT main effect of Group")
                print(
                    f"    → {' and '.join(comparison_groups)} groups differ significantly overall across fibrosis levels")
            else:
                print(f"No significant main effect of Group (p = {group_p:.4f})")
                print(f"    → No overall difference between {' and '.join(comparison_groups)} groups")

            if bin_p < 0.05:
                sig_level = "***" if bin_p < 0.001 else "**" if bin_p < 0.01 else "*"
                print(f"{sig_level} SIGNIFICANT main effect of Fibrosis Bin")
                print(f"    → {metric.replace('_', ' ')} changes significantly with fibrosis level")
            else:
                print(f"No significant main effect of Fibrosis Bin (p = {bin_p:.4f})")

            if interaction_p < 0.05:
                sig_level = "***" if interaction_p < 0.001 else "**" if interaction_p < 0.01 else "*"
                print(f"{sig_level} SIGNIFICANT interaction effect")
                print(f"    → Groups respond DIFFERENTLY to increasing fibrosis")
                print(f"    → This suggests distinct pathophysiological mechanisms")
            else:
                print(f"No significant interaction (p = {interaction_p:.4f})")
                print(f"    → Groups respond similarly to fibrosis progression")

            print(f"\nModel Fit: R² = {model.rsquared:.4f} (explains {model.rsquared * 100:.1f}% of variance)")

            # Perform post-hoc analysis for pairwise comparisons within each bin
            if group_p < 0.05 or interaction_p < 0.05:
                print(f"\nPost-hoc Analysis: Pairwise Group Comparisons on {metric} within each Fibrosis Bin {bin_width_percent}%")
                print("." * 40)

                post_hoc_results = []

                # Generate all pairwise comparisons
                from itertools import combinations
                group_pairs = list(combinations(comparison_groups, 2))

                for bin_name in valid_bins:
                    bin_data = analysis_data[analysis_data['fibrosis_bin'] == bin_name]

                    print(f"\nFibrosis Bin: {bin_name}")

                    for group1, group2 in group_pairs:
                        group1_data = bin_data[bin_data['group'] == group1][metric]
                        group2_data = bin_data[bin_data['group'] == group2][metric]

                        if len(group1_data) >= 2 and len(group2_data) >= 2:
                            # Perform t-test within this bin
                            t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)

                            # Calculate effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) +
                                                  (len(group2_data) - 1) * np.var(group2_data, ddof=1)) /
                                                 (len(group1_data) + len(group2_data) - 2))
                            cohens_d = (np.mean(group1_data) - np.mean(
                                group2_data)) / pooled_std if pooled_std > 0 else 0

                            # Apply Bonferroni correction for multiple comparisons
                            n_comparisons = len(valid_bins) * len(group_pairs)
                            p_corrected = min(p_value * n_comparisons, 1.0)

                            post_hoc_results.append({
                                'fibrosis_bin': bin_name,
                                'group1': group1,
                                'group1_n': len(group1_data),
                                'group1_mean': np.mean(group1_data),
                                'group1_std': np.std(group1_data, ddof=1),
                                'group2': group2,
                                'group2_n': len(group2_data),
                                'group2_mean': np.mean(group2_data),
                                'group2_std': np.std(group2_data, ddof=1),
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'p_corrected': p_corrected,
                                'mean_difference': np.mean(group1_data) - np.mean(group2_data),
                                'cohens_d': cohens_d,
                                'significant': p_corrected < 0.05
                            })

                            # Display results
                            print(f"  {group1} vs {group2}:")
                            print(
                                f"    {group1}: {np.mean(group1_data):.4f} ± {np.std(group1_data, ddof=1):.4f} (n={len(group1_data)})")
                            print(
                                f"    {group2}: {np.mean(group2_data):.4f} ± {np.std(group2_data, ddof=1):.4f} (n={len(group2_data)})")
                            print(f"    t = {t_stat:.3f}, p = {p_value:.4f}, p_corrected = {p_corrected:.4f}")
                            print(f"    Cohen's d = {cohens_d:.3f}")

                            if p_corrected < 0.001:
                                print("    *** HIGHLY SIGNIFICANT after correction")
                            elif p_corrected < 0.01:
                                print("    ** SIGNIFICANT after correction")
                            elif p_corrected < 0.05:
                                print("    * SIGNIFICANT after correction")
                            else:
                                print("    Not significant after correction")

                results['post_hoc_results'].append({
                    'metric': metric,
                    'bin_comparisons': post_hoc_results
                })

                # Summary of post-hoc results
                significant_comparisons = [r for r in post_hoc_results if r['significant']]
                print(
                    f"\nPost-hoc Summary: {len(significant_comparisons)}/{len(post_hoc_results)} comparisons show significant group differences")

                if significant_comparisons:
                    print("Significant group differences found in:")
                    for result in significant_comparisons:
                        direction = "higher" if result['mean_difference'] > 0 else "lower"
                        effect_size = "large" if abs(result['cohens_d']) > 0.8 else "medium" if abs(
                            result['cohens_d']) > 0.5 else "small"
                        print(
                            f"  {result['fibrosis_bin']}: {result['group1']} {direction} than {result['group2']} ({effect_size} effect)")

        except Exception as e:
            print(f"Error in two-way ANOVA for {metric}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return results

def perform_stratified_regression_analysis(df, metrics, comparison_groups=None, exclude_groups=None):
    """
    Compare linear regressions between specified groups

    Args:
        df: DataFrame with connexin data
        metrics: List of metrics to analyze
        comparison_groups: List of groups to compare (e.g., ['infarct', 'radiation'] or ['chronicallyoccluded', 'infarct'])
                          If None, will auto-detect non-excluded groups
        exclude_groups: List of groups to exclude (e.g., ['healthy', 'control'])
    """
    print('\nPerforming stratified regression analysis.')
    print('-' * 40)

    # Filter out excluded groups
    df_filtered = df[~df['group'].isin(exclude_groups)].copy()

    if len(comparison_groups) < 2:
        print(f"Error: Need at least 2 groups for comparison. Found: {comparison_groups}")
        return {}

    results = {
        'comparison_groups': comparison_groups,
        'excluded_groups': exclude_groups,
        'regression_results': [],
        'slope_comparisons': []
    }

    for metric in metrics:
        print(f"\nRegression Analysis Between: {metric.replace('_', ' ').title()} vs Fibrosis %")
        print(f"Groups: {' vs '.join(comparison_groups)}")
        print(f"{'.' * 40}")

        # Prepare data
        analysis_data = df_filtered[['Fibrosis Percentage', metric, 'group']].dropna()

        if len(analysis_data) < 10:
            print(f"Insufficient data for {metric} (n={len(analysis_data)})")
            continue

        metric_results = {
            'metric': metric,
            'comparison_groups': comparison_groups,
            'overall_regression': {},
            'group_regressions': {},
            'slope_comparison': {}
        }

        # Overall regression (all data combined)
        x_overall = analysis_data['Fibrosis Percentage']
        y_overall = analysis_data[metric]
        slope, intercept, r_value, p_value, std_err = linregress(x_overall, y_overall)

        metric_results['overall_regression'] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'n': len(analysis_data)
        }

        print(f"Overall regression (all groups combined):")
        print(f"  {metric} = {slope:.6f} × Fibrosis% + {intercept:.6f}")
        print(f"  R² = {r_value ** 2:.4f}, p = {p_value:.4f}, n = {len(analysis_data)}")

        # Separate regressions for each group
        group_slopes = {}
        for group in comparison_groups:
            group_data = analysis_data[analysis_data['group'] == group]

            if len(group_data) < 5:
                print(f"Insufficient data for {group} group (n={len(group_data)})")
                continue

            x_group = group_data['Fibrosis Percentage']
            y_group = group_data[metric]

            slope_g, intercept_g, r_value_g, p_value_g, std_err_g = linregress(x_group, y_group)
            group_slopes[group] = slope_g

            metric_results['group_regressions'][group] = {
                'slope': slope_g,
                'intercept': intercept_g,
                'r_value': r_value_g,
                'r_squared': r_value_g ** 2,
                'p_value': p_value_g,
                'std_err': std_err_g,
                'n': len(group_data)
            }

            print(f"\n{group.title()} group regression:")
            print(f"  {metric} = {slope_g:.6f} × Fibrosis% + {intercept_g:.6f}")
            print(f"  R² = {r_value_g ** 2:.4f}, p = {p_value_g:.4f}, n = {len(group_data)}")

        # ANCOVA to test if slopes are significantly different
        if len(group_slopes) >= 2:
            try:
                # Clean column names for formula - replace any problematic characters
                clean_metric_name = metric.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')

                # Create a copy of the data with clean column names for the formula
                formula_data = analysis_data.copy()
                formula_data[clean_metric_name] = formula_data[metric]
                formula_data['Fibrosis_Percentage'] = formula_data['Fibrosis Percentage']

                # Test for different slopes (interaction model)
                formula_interaction = f'{clean_metric_name} ~ C(group) * Q("Fibrosis_Percentage")'
                model_interaction = ols(formula_interaction, data=formula_data).fit()

                # Test for parallel slopes (additive model)
                formula_parallel = f'{clean_metric_name} ~ C(group) + Q("Fibrosis_Percentage")'
                model_parallel = ols(formula_parallel, data=formula_data).fit()

                # Compare models to test slope difference
                anova_comparison = anova_lm(model_parallel, model_interaction)
                slope_diff_p = anova_comparison['Pr(>F)'].iloc[1] if len(anova_comparison) > 1 else 1.0

                # Calculate slope differences for all pairs
                slope_differences = {}
                if len(comparison_groups) >= 2:
                    from itertools import combinations
                    for group1, group2 in combinations(comparison_groups, 2):
                        if group1 in group_slopes and group2 in group_slopes:
                            slope_differences[f"{group1}_vs_{group2}"] = group_slopes[group1] - group_slopes[group2]

                metric_results['slope_comparison'] = {
                    'slopes_different': slope_diff_p < 0.05,
                    'p_value': slope_diff_p,
                    'group_slopes': group_slopes,
                    'slope_differences': slope_differences,
                    'interaction_model': model_interaction,
                    'parallel_model': model_parallel
                }

                print(f"\nSlope Comparison (ANCOVA):")
                for group, slope in group_slopes.items():
                    print(f"  {group.title()} slope: {slope:.6f}")

                if len(slope_differences) > 0:
                    print(f"\nSlope differences:")
                    for comparison, diff in slope_differences.items():
                        print(f"  {comparison.replace('_vs_', ' vs ')}: {diff:.6f}")

                print(f"  p-value for slope difference: {slope_diff_p:.4f}")

                if slope_diff_p < 0.001:
                    print("  *** HIGHLY SIGNIFICANT difference in slopes")
                    print("      → Groups respond very differently to fibrosis progression")
                elif slope_diff_p < 0.01:
                    print("  ** SIGNIFICANT difference in slopes")
                    print("      → Groups respond differently to fibrosis progression")
                elif slope_diff_p < 0.05:
                    print("  * SIGNIFICANT difference in slopes")
                    print("      → Groups respond differently to fibrosis progression")
                else:
                    print("  No significant difference in slopes (parallel regression lines)")
                    print("      → Groups respond similarly to fibrosis progression")

                # Model comparison statistics
                print(f"\nModel Comparison:")
                print(f"  Interaction model R² = {model_interaction.rsquared:.4f}")
                print(f"  Parallel model R² = {model_parallel.rsquared:.4f}")
                print(f"  Improvement with interaction = {model_interaction.rsquared - model_parallel.rsquared:.4f}")

            except Exception as e:
                print(f"Error in slope comparison: {str(e)}")

        results['regression_results'].append(metric_results)

    return results


def create_stratified_analysis_plots(df, anova_results, regression_results, bin_width_percent=10,
                                     comparison_groups=None, exclude_groups=None):
    """
    Create comprehensive plots for the stratified analysis
    """
    # Set default exclusions
    if exclude_groups is None:
        exclude_groups = ['healthy', 'control']

    # Filter out excluded groups
    df_filtered = df[~df['group'].isin(exclude_groups)].copy()

    # Determine comparison groups
    if comparison_groups is None:
        available_groups = df_filtered['group'].unique()
        comparison_groups = list(available_groups)
    else:
        df_filtered = df_filtered[df_filtered['group'].isin(comparison_groups)].copy()

    df_binned, _ = create_fibrosis_bins(df_filtered, bin_width_percent)

    metrics = [result['metric'] for result in anova_results.get('anova_results', [])]
    n_metrics = len(metrics)

    if n_metrics == 0:
        print("No metrics to plot")
        return None

    # Create subplot layout
    fig, axes = plt.subplots(2, n_metrics, figsize=(5 * n_metrics, 10))
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Stratified Analysis: {" vs ".join(comparison_groups)} by Fibrosis Level',
                 fontsize=16, fontweight='bold')

    # Create a color palette for all groups
    colors = plt.cm.Set1(np.linspace(0, 1, len(comparison_groups)))
    color_dict = {group: colors[i] for i, group in enumerate(comparison_groups)}

    for i, metric in enumerate(metrics):
        # Top row: Box plots by fibrosis bin
        ax1 = axes[0, i]
        analysis_data = df_binned[['fibrosis_bin', metric, 'group']].dropna()

        if len(analysis_data) > 0:
            sns.boxplot(data=analysis_data, x='fibrosis_bin', y=metric, hue='group',
                        ax=ax1, palette=color_dict)
            ax1.set_title(f'Box Plot: {metric.replace("_", " ").title()}')
            ax1.set_xlabel('Fibrosis Bin (%)')
            ax1.tick_params(axis='x', rotation=45)

            # Add sample sizes
            bin_order = sorted(analysis_data['fibrosis_bin'].unique())
            for j, bin_name in enumerate(bin_order):
                for k, group in enumerate(comparison_groups):
                    n = len(analysis_data[(analysis_data['fibrosis_bin'] == bin_name) &
                                          (analysis_data['group'] == group)])
                    if n > 0:
                        ax1.text(j + (k - len(comparison_groups) / 2 + 0.5) * 0.3, ax1.get_ylim()[1] * 0.95, f'n={n}',
                                 ha='center', va='top', fontsize=8, color=color_dict[group])

        # Bottom row: Regression plots
        ax2 = axes[1, i]
        reg_data = df_filtered[['Fibrosis Percentage', metric, 'group']].dropna()

        if len(reg_data) > 0:
            # Plot scatter points
            for group in comparison_groups:
                group_data = reg_data[reg_data['group'] == group]
                if len(group_data) > 0:
                    ax2.scatter(group_data['Fibrosis Percentage'], group_data[metric],
                                c=[color_dict[group]], label=group.title(), alpha=0.6, s=30)

            # Plot regression lines
            reg_result = None
            for result in regression_results.get('regression_results', []):
                if result['metric'] == metric:
                    reg_result = result
                    break

            if reg_result:
                x_range = np.linspace(reg_data['Fibrosis Percentage'].min(),
                                      reg_data['Fibrosis Percentage'].max(), 100)

                for group in comparison_groups:
                    if group in reg_result['group_regressions']:
                        slope = reg_result['group_regressions'][group]['slope']
                        intercept = reg_result['group_regressions'][group]['intercept']
                        r_squared = reg_result['group_regressions'][group]['r_squared']
                        p_value = reg_result['group_regressions'][group]['p_value']

                        y_pred = slope * x_range + intercept
                        ax2.plot(x_range, y_pred, color=color_dict[group], linewidth=2,
                                 label=f'{group.title()}: R²={r_squared:.3f}, p={p_value:.3f}')

            ax2.set_xlabel('Fibrosis Percentage (%)')
            ax2.set_ylabel(metric.replace('_', ' ').title())
            ax2.set_title(f'Regression: {metric.replace("_", " ").title()}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stratified_analysis_plots.png', dpi=300, bbox_inches='tight')
    print(f"\nStratified analysis plots saved as 'stratified_analysis_plots.png'")
    plt.show()

    return fig

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
    print("\nSAMPLE SIZES BY GROUP:")
    print("-" * 40)
    
    group_counts = df['group_scar'].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"{group}: {count} samples")

    return group_counts

def print_anova_results(anova_results):
    """Print ANOVA results in a formatted way"""
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
    print("\nT-TEST for Grouped Comparisons:")
    print("-" * 40)
    
    for comparison in ttest_df['Comparison'].unique():
        print(f"\n{comparison.upper().replace('_', ' ')}:")
        print("." * 20)
        
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
    print("\nLegacy (By Histopathology Defined Scar Location) Key Findings:")
    print("-" * 40)
    
    # Find significant results
    significant_ttests = ttest_df[ttest_df['p_value'] < 0.05]
    significant_anovas = [r for r in anova_results if r['p_value'] < 0.05]
    
    print(f"\n\tSIGNIFICANT FINDINGS ({len(significant_ttests)} significant comparisons found):")
    
    if len(significant_ttests) > 0:
        for _, row in significant_ttests.iterrows():
            direction = "higher" if row['mean_difference'] > 0 else "lower"
            print(f"   • {row['Comparison']}: {row['group1']} shows significantly {direction}")
            print(f"     {row['Metric'].replace('_', ' ')} than {row['group2']} (p = {row['p_value']:.4f})")
    else:
        print("   • No significant differences found in pairwise comparisons")
    
    if len(significant_anovas) > 0:
        print(f"\n\tSIGNIFICANT ANOVA RESULTS:")
        for result in significant_anovas:
            print(f"   • {result['Metric'].replace('_', ' ')}: F = {result['F_statistic']:.3f}, p = {result['p_value']:.4f}")
    
    # Identify largest effect sizes
    print(f"\n\tLARGEST EFFECT SIZES:")
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

def main_analysis(filepath, comparison_groups=None, exclude_groups=None, metrics= ['connexin_area_per_annotation', 'connexin_count_per_annotation']):
    """Main analysis function"""
    print("*" * 80)
    print("CONNEXIN STATISTICAL ANALYSIS TOOLKIT")
    print("3 Parts: ")
    print("    1. Legacy Analysis by Pathology Defined Scar Location")
    print("    2. Connexin Analysis grouped by Fibrosis Percentage")
    print("    3. Connexin Analysis via Linear Regression")
    print("*" * 80)

    print('')
    print("=" * 80)
    print("1. CONNEXIN ANALYSIS BY PATHOLOGY DEFINED SCAR LOCATION")
    print("=" * 80)
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
        print('location column (apex, mid, base designation) empty.')
    
    # Get comparison groups dynamically
    comparisons = get_comparison_groups(df)
    
    # Print sample sizes
    sample_sizes = print_sample_sizes(df)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(df, comparisons, metrics)
    
    # Display summary statistics table
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    
    # Pivot table for better readability
    for metric in metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        metric_data = summary_stats[summary_stats['Metric'] == metric]
        
        for _, row in metric_data.iterrows():
            print(f"  {row['Group']}: {row['Mean']:.6f} ± {row['Std']:.6f} (n={row['N']})")

    ## STARTING LEGACY STATISTICAL ANALYSIS ############################################################################
    # Perform ANOVA for healthy tissue comparison
    print("\nANOVA for Healthy Tissue Comparison")
    print("-" * 40)
    healthy_groups = [group for group in df['group_scar'].unique() if '_healthy' in group]
    
    if len(healthy_groups) >= 2:
        anova_results = perform_anova_analysis(df, healthy_groups, metrics)
        print_anova_results(anova_results)
    else:
        print("\nInsufficient healthy tissue groups for ANOVA analysis")
        anova_results = []
    
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

    else:
        print("No suitable pairs found for t-test comparisons")
        ttest_results = pd.DataFrame()
    
    # Correlation analysis
    correlation_pairs = [
        ('Fibrosis Percentage', 'connexin_area_per_annotation'),
        ('Fibrosis Percentage', 'connexin_count_per_annotation'),
        ('Fibrosis Percentage', 'connexin_area_per_cell'),
        ('Fibrosis Percentage', 'mean_plaque_size')]#,
        #('Fibrosis Percentage', 'cell_density'),
        #('cell_density', 'connexin_area_per_cell')]
    
    # Only include correlations for variables that exist in the dataset
    available_correlations = [(var1, var2) for var1, var2 in correlation_pairs 
                            if var1 in df.columns and var2 in df.columns]
    
    if available_correlations:
        correlation_results = analyze_correlations(df, available_correlations)
        
        print("\nFibrosis Percentage Correlations:")
        print("-" * 40)
        for _, row in correlation_results.iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"{row['Variable_1']} vs {row['Variable_2']}: r = {row['Correlation_r']:.4f}, p = {row['p_value']:.4f} {significance} (n={row['N']})")

    # Identify key findings
    identify_key_findings(ttest_results, anova_results)

    # Create visualizations
    #create_visualizations(df)

    ## STARTING GROUPED FIBROSIS ANALYSIS ##############################################################################
    print('')
    print('=' * 80)
    print("2. CONNEXIN ANALYSIS GROUPED BY FIBROSIS PERCENTAGE")
    print("=" * 80)
    # Set default exclusions if not provided
    if exclude_groups is None:
        exclude_groups = ['healthy', 'control']

    # Determine comparison groups
    if comparison_groups is None:
        # Auto-detect available groups (excluding specified groups)
        available_groups = df[~df['group'].isin(exclude_groups)]['group'].unique()
        comparison_groups = list(available_groups)
        print(f"Auto-detected comparison groups: {comparison_groups}")
    else:
        print(f"Using specified comparison groups: {comparison_groups}")

    print(f"Groups to exclude from stratified analysis: {exclude_groups}")

    stratified_anova_results_10 = perform_stratified_two_way_anova(
        df, metrics, bin_width_percent=10,
        comparison_groups=comparison_groups, exclude_groups=exclude_groups
    )
    stratified_anova_results_20 = perform_stratified_two_way_anova(
        df, metrics, bin_width_percent=20,
        comparison_groups=comparison_groups, exclude_groups=exclude_groups
    )
    stratified_anova_results_25 = perform_stratified_two_way_anova(
        df, metrics, bin_width_percent=25,
        comparison_groups=comparison_groups, exclude_groups=exclude_groups
    )
    stratified_anova_results_33 = perform_stratified_two_way_anova(
        df, metrics, bin_width_percent=33,
        comparison_groups=comparison_groups, exclude_groups=exclude_groups
    )

    # Linear regression analysis
    print('')
    print('=' * 80)
    print("3. LINEAR REGRESSION ANALYSIS")
    print("=" * 80)

    regression_results = perform_stratified_regression_analysis(
        df, metrics, comparison_groups=comparison_groups, exclude_groups=exclude_groups
    )

    significant_anova_count = 0
    significant_regression_count = 0
    different_slopes_count = 0

    #for result in anova_results['anova_results']:
    #    anova_table = result['anova_table']
    #    group_p = anova_table.loc['C(group)', 'PR(>F)']
    #    bin_p = anova_table.loc['C(fibrosis_bin)', 'PR(>F)']
    #    interaction_p = anova_table.loc['C(group):C(fibrosis_bin)', 'PR(>F)']

    #    if group_p < 0.05 or bin_p < 0.05 or interaction_p < 0.05:
    #        significant_anova_count += 1

    for result in regression_results['regression_results']:
        # Check if either group has significant regression
        for group in ['infarct', 'radiation']:
            if group in result['group_regressions']:
                if result['group_regressions'][group]['p_value'] < 0.05:
                    significant_regression_count += 1
                    break

        # Check if slopes are different
        if 'ancova' in result and 'slopes_different' in result['ancova']:
            if result['ancova']['slopes_different']:
                different_slopes_count += 1

    print(f"\nKEY FINDINGS:")
    print(f"• {significant_anova_count}/{len(metrics)} metrics showed significant differences in two-way ANOVA")
    print(f"• {significant_regression_count}/{len(metrics)} metrics showed significant regression with fibrosis")
    print(f"• {different_slopes_count}/{len(metrics)} metrics showed significantly different slopes between groups")

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
        comparison_groups = None
        exclude_groups = ['healthy', 'control']
        metrics = ['connexin_area_per_annotation', 'connexin_count_per_annotation', 'connexin_area_per_cell',
                   'mean_plaque_size']
        results = main_analysis(filepath, comparison_groups=comparison_groups, exclude_groups=exclude_groups, metrics=metrics)
        print("\nAnalysis completed successfully!")
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
