# Pro-inflammatory Lipid Index Analysis
# Author: [Luan Vu - UTSA and Joan Cook-Mills - IU]
# Date: December 2024

"""
This script performs a comprehensive analysis of lipid profiles and their relationship
to clinical outcomes in different patient clusters. The analysis is divided into three
main stages:

1. Data Preparation and Initial Analysis
2. Statistical Comparison with Control Group
3. Pro-inflammatory Lipid Index Calculation and Correlation Analysis
"""

# Import required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the data, including basic data cleaning and validation.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    df = pd.read_csv(file_path)
    
    # Exclude comment columns
    columns_to_exclude = ['aT (uM),comment', 'gT (uM), comment']
    df = df.drop(columns=columns_to_exclude, errors='ignore')

    # Create cluster labels
    df['cluster_label'] = 'C' + df['kmcluster'].astype(str)
    
    # Convert binary outcomes to numeric if needed
    df['Wheeze_1yr'] = pd.to_numeric(df['Wheeze_1yr'], errors='coerce')
    df['AD_1yr'] = pd.to_numeric(df['AD_1yr'], errors='coerce')
    
    return df

# =================
# Stage 1: Data Preparation and Initial Analysis
# =================

def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing data and create summary.
    """
    missing_data = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    return missing_data[missing_data['Missing Values'] > 0]


def analyze_tocopherol_distribution(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze tocopherol distributions by cluster and clinical outcomes.
    """
    # Basic statistics by cluster
    cluster_stats = df.groupby('cluster_label').agg({
        'alpha-tocopherol': ['count', 'mean', 'std', 'min', 'max'],
        'gamma-tocopherol': ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    
    # Distribution by clinical outcomes
    outcome_stats = {
        'Wheeze': df.groupby('Wheeze_1yr')[['alpha-tocopherol', 'gamma-tocopherol']].describe(),
        'AD': df.groupby('AD_1yr')[['alpha-tocopherol', 'gamma-tocopherol']].describe()
    }
    
    return {'cluster': cluster_stats, 'outcomes': outcome_stats}

def create_stage1_plots(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create visualizations for Stage 1 analysis.
    """
    # Distribution plots
    fig_dist = make_subplots(rows=1, cols=2,
                            subplot_titles=('Alpha-Tocopherol Distribution',
                                          'Gamma-Tocopherol Distribution'))
    
    for i, col in enumerate(['alpha-tocopherol', 'gamma-tocopherol'], 1):
        fig_dist.add_trace(
            go.Violin(x=df['cluster_label'],
                     y=df[col],
                     name=col,
                     box_visible=True,
                     meanline_visible=True),
            row=1, col=i
        )
        
    # Update y-axis titles
    fig_dist.update_yaxes(title_text="Concentration (µM)", row=1, col=1)
    fig_dist.update_yaxes(title_text="Concentration (µM)", row=1, col=2)
    fig_dist.update_xaxes(title_text="Cluster", row=1, col=1)
    fig_dist.update_xaxes(title_text="Cluster", row=1, col=2)
    
    # Outcome distribution
    fig_outcomes = go.Figure()
    for outcome in ['Wheeze_1yr', 'AD_1yr']:
        fig_outcomes.add_trace(
            go.Bar(name=outcome,
                  x=['Yes', 'No'],
                  y=[sum(df[outcome] == 1), sum(df[outcome] == 0)])
        )
    fig_outcomes.update_layout(
        yaxis_title="Number of Patients",
        xaxis_title="Clinical Outcome",
        title="Distribution of Clinical Outcomes"
    )
    
    return {'distributions': fig_dist, 'outcomes': fig_outcomes}

# =================
# Stage 2: Statistical Analysis
# =================

def compare_with_control(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Perform statistical comparison with control group (C6).
    """
    control_data = df[df['cluster_label'] == 'C6'][variable].dropna()
    results = []
    
    for cluster in ['C1', 'C2', 'C3', 'C4', 'C5', 'C7']:
        test_data = df[df['cluster_label'] == cluster][variable].dropna()
        
        # T-test
        t_stat, p_val = stats.ttest_ind(test_data, control_data)
        
        # Effect size
        n1, n2 = len(test_data), len(control_data)
        pooled_std = np.sqrt(((n1 - 1) * test_data.std()**2 + 
                            (n2 - 1) * control_data.std()**2) / (n1 + n2 - 2))
        cohens_d = (test_data.mean() - control_data.mean()) / pooled_std
        
        results.append({
            'cluster': cluster,
            'mean_diff': test_data.mean() - control_data.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'adjusted_p': min(p_val * 6, 1),  # Bonferroni correction
            'cohens_d': cohens_d
        })
    
    return pd.DataFrame(results)

def create_stage2_plots(df: pd.DataFrame, stats_results: Dict) -> go.Figure:
    """
    Create visualizations for Stage 2 analysis.
    """
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Alpha-Tocopherol by Cluster',
                                      'Gamma-Tocopherol by Cluster',
                                      'Statistical Significance',
                                      'Effect Sizes'),
                        vertical_spacing=0.5,
                        horizontal_spacing=0.5)
    
    # Add violin plots
    for i, col in enumerate(['alpha-tocopherol', 'gamma-tocopherol'], 1):
        fig.add_trace(
            go.Violin(x=df['cluster_label'],
                     y=df[col],
                     name=col,
                     box_visible=True,
                     meanline_visible=True,
                     points="all"),
            row=1, col=i
        )
    
    # Add significance and effect size plots
    fig.add_trace(
        go.Bar(x=stats_results['alpha']['cluster'],
               y=-np.log10(stats_results['alpha']['adjusted_p']),
               name='Alpha -log10(p)'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=stats_results['alpha']['cluster'],
               y=stats_results['alpha']['cohens_d'],
               name="Cohen's d"),
        row=2, col=2
    )
    
    # Update axis titles
    fig.update_yaxes(title_text="Concentration (µM)", row=1, col=1)
    fig.update_yaxes(title_text="Concentration (µM)", row=1, col=2)
    fig.update_yaxes(title_text="-log10(adjusted p-value)", row=2, col=1)
    fig.update_yaxes(title_text="Effect Size (Cohen's d)", row=2, col=2)
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Cluster", row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=1, col=2)
    fig.update_xaxes(title_text="Cluster", row=2, col=1)
    fig.update_xaxes(title_text="Cluster", row=2, col=2)
    
    return fig
# =================
# Stage 3: Proinflammatory Lipid Index
# =================

def calculate_tocopherol_fc(df: pd.DataFrame) -> Dict:
    """
    Calculate log2 fold changes for tocopherols relative to cluster 6.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing tocopherol measurements
        
    Returns:
    --------
    Dict
        Dictionary containing log2FC values for each cluster
    """
    cluster_means = df.groupby('kmcluster').agg({
        'alpha-tocopherol': 'mean',
        'gamma-tocopherol': 'mean'
    })
    
    # Get C6 values for reference
    c6_alpha = cluster_means.loc[6, 'alpha-tocopherol']
    c6_gamma = cluster_means.loc[6, 'gamma-tocopherol']
    
    # Calculate log2FC for each cluster
    log2fc = {}
    for cluster in range(1, 8):  # Clusters 1-7
        if cluster in cluster_means.index:
            log2fc[cluster] = {
                'alpha_log2fc': np.log2(cluster_means.loc[cluster, 'alpha-tocopherol'] / c6_alpha),
                'gamma_log2fc': np.log2(cluster_means.loc[cluster, 'gamma-tocopherol'] / c6_gamma)
            }
    
    return log2fc

def calculate_proinflammatory_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pro-inflammatory lipid index for each cluster.
    Includes log2FC for all components including tocopherols.
    """
    # Get tocopherol fold changes
    tocopherol_fc = calculate_tocopherol_fc(df)
    
    # Pre-calculated log2FC values
    provided_data = {
        'cluster#': [1, 2, 3, 4, 5, 6, 7, 8],
        'βGlcCers (mean log2FC)': [1.33, 1.34, 1.60, 0.79, 0.90, 0, 0.76, 1.55],
        'βGalCers (mean log2FC)': [2.74, 3.30, 1.59, 0.93, 1.04, 0, 1.36, 2.74],
        'sphingomyelins (mean log2FC)': [0.75, None, 0.69, None, 0.67, 0, 0.83, 0.71],
        'Lipoxin A4 (log2FC)': [None, None, None, None, None, None, -2.34, None],
        'Resolvin D2 (log2FC)': [-2.39, None, None, None, None, None, -0.81, None],
        'alpha-tocopherol (log2FC)': [None] * 8,  # Initialize with None
        'gamma-tocopherol (log2FC)': [None] * 8   # Initialize with None
    }
    
    index_df = pd.DataFrame(provided_data)
    
    # Add tocopherol log2FC values to the table
    for cluster in range(1, 8):  # Only calculate for clusters 1-7
        if cluster in tocopherol_fc:
            index_df.loc[index_df['cluster#'] == cluster, 'alpha-tocopherol (log2FC)'] = \
                round(tocopherol_fc[cluster]['alpha_log2fc'], 2)
            index_df.loc[index_df['cluster#'] == cluster, 'gamma-tocopherol (log2FC)'] = \
                round(tocopherol_fc[cluster]['gamma_log2fc'], 2)
            
            # Calculate pro-inflammatory index
            row_values = index_df.loc[index_df['cluster#'] == cluster].iloc[0]
            index_sum = 0
            
            # Add pre-calculated values
            for col in ['βGlcCers (mean log2FC)', 'βGalCers (mean log2FC)', 
                       'sphingomyelins (mean log2FC)', 'Lipoxin A4 (log2FC)', 
                       'Resolvin D2 (log2FC)']:
                if pd.notnull(row_values[col]):
                    index_sum += row_values[col]
            
            # Add tocopherol contributions
            index_sum += (-1 * tocopherol_fc[cluster]['alpha_log2fc'])  # Inverse alpha
            index_sum += tocopherol_fc[cluster]['gamma_log2fc']  # Direct gamma
            
            # Store the final index
            index_df.loc[index_df['cluster#'] == cluster, 'pro-inflammatory lipid index'] = round(index_sum, 2)
    
    # Reorder columns to put tocopherols before the final index
    column_order = [
        'cluster#', 
        'βGlcCers (mean log2FC)',
        'βGalCers (mean log2FC)', 
        'sphingomyelins (mean log2FC)',
        'Lipoxin A4 (log2FC)',
        'Resolvin D2 (log2FC)',
        'alpha-tocopherol (log2FC)',
        'gamma-tocopherol (log2FC)',
        'pro-inflammatory lipid index'
    ]
    
    return index_df[column_order]

# First define helper functions
def calculate_spearman_ci(x: pd.Series, y: pd.Series, n_bootstrap: int = 5000, 
                      confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for Spearman correlation using bootstrapping.
    """
    rhos = []
    n = len(x)
    indices = np.arange(n)
    
    # Ensure x and y are numpy arrays
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    try:
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(indices, size=n, replace=True)
            rho, _ = stats.spearmanr(x_arr[boot_indices], y_arr[boot_indices])
            if not np.isnan(rho):  # Only append valid correlations
                rhos.append(rho)
        
        if len(rhos) > 0:  # Only calculate CI if we have valid correlations
            ci_lower = np.percentile(rhos, (1 - confidence_level) * 100 / 2)
            ci_upper = np.percentile(rhos, 100 - (1 - confidence_level) * 100 / 2)
            return ci_lower, ci_upper
    except:
        pass
    
    return np.nan, np.nan  # Return nan if calculation fails


def perform_correlation_analysis(df: pd.DataFrame, 
                               proinflammatory_df: pd.DataFrame) -> Dict[str, Tuple]:
    """
    Perform correlation analysis between pro-inflammatory index and outcomes.
    """
    # Calculate percentage for each cluster
    cluster_outcomes = df.groupby('kmcluster').agg({
        'Wheeze_1yr': lambda x: (x == 1).mean() * 100,
        'AD_1yr': lambda x: (x == 1).mean() * 100
    })
    
    x_data = proinflammatory_df['pro-inflammatory lipid index'][:7]  # Exclude cluster 8
    wheeze_data = cluster_outcomes['Wheeze_1yr']
    ad_data = cluster_outcomes['AD_1yr']

    # Calculate correlations and CIs
    wheeze_corr = stats.spearmanr(x_data, wheeze_data)
    ad_corr = stats.spearmanr(x_data, ad_data)
    
    wheeze_ci = calculate_spearman_ci(x_data, wheeze_data)
    ad_ci = calculate_spearman_ci(x_data, ad_data)
    
    return {
        'wheeze': (wheeze_corr, wheeze_ci),
        'ad': (ad_corr, ad_ci)
    }

def create_correlation_plots(df: pd.DataFrame, 
                           proinflammatory_df: pd.DataFrame,
                           correlation_results: Dict) -> go.Figure:
    """
    Create correlation plots with annotations outside plot area.
    """
    # Calculate percentage for each cluster
    cluster_outcomes = df.groupby('kmcluster').agg({
        'Wheeze_1yr': lambda x: (x == 1).mean() * 100,
        'AD_1yr': lambda x: (x == 1).mean() * 100
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Pro-inflammatory Index vs Wheeze',
            'Pro-inflammatory Index vs AD'
        ),
        horizontal_spacing=0.2
    )
    
    # Prepare data
    x_data = proinflammatory_df['pro-inflammatory lipid index'][:7]
    wheeze_data = cluster_outcomes['Wheeze_1yr']
    ad_data = cluster_outcomes['AD_1yr']
    
    # Calculate regression lines
    wheeze_coeffs = np.polyfit(x_data, wheeze_data, 1)
    ad_coeffs = np.polyfit(x_data, ad_data, 1)
    
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    wheeze_line = wheeze_coeffs[0] * x_range + wheeze_coeffs[1]
    ad_line = ad_coeffs[0] * x_range + ad_coeffs[1]
    
    # Get correlation results
    wheeze_stats = correlation_results['wheeze']
    ad_stats = correlation_results['ad']
    
    # Wheeze plot
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=wheeze_data,
            mode='markers+text',
            name='Clusters',
            text=[f'C{i}' for i in range(1, 8)],
            textposition="top center",
            marker=dict(size=12, color='blue'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=wheeze_line,
            mode='lines',
            name='Wheeze',
            line=dict(color='blue', dash='dot')
        ),
        row=1, col=1
    )
    
    # AD plot
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=ad_data,
            mode='markers+text',
            name='Clusters',
            text=[f'C{i}' for i in range(1, 8)],
            textposition="top center",
            marker=dict(size=12, color='red'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=ad_line,
            mode='lines',
            name='AD',
            line=dict(color='red', dash='dot')
        ),
        row=1, col=2
    )
    
    # Add annotations for correlation coefficients
    # Wheeze annotation
    wheeze_annotation = f"ρ = {wheeze_stats[0][0]:.2f}<br>p = {wheeze_stats[0][1]:.3f}"
    if not np.isnan(wheeze_stats[1][0]) and not np.isnan(wheeze_stats[1][1]):
        wheeze_annotation += f"<br>95% CI: [{wheeze_stats[1][0]:.2f}, {wheeze_stats[1][1]:.2f}]"
    
    fig.add_annotation(
        x=0.25,  # Position outside the plot
        y=-0.22,   # Position above the plot
        xref="paper",
        yref="paper",
        text=wheeze_annotation,
        showarrow=False,
        font=dict(size=12, color="blue"),
        align="left",
        bordercolor="blue",
        borderwidth=1,
        borderpad=4,
        xanchor="center",
        yanchor="top"
    )
    
    # AD annotation
    ad_annotation = f"ρ = {ad_stats[0][0]:.2f}<br>p = {ad_stats[0][1]:.3f}"
    if not np.isnan(ad_stats[1][0]) and not np.isnan(ad_stats[1][1]):
        ad_annotation += f"<br>95% CI: [{ad_stats[1][0]:.2f}, {ad_stats[1][1]:.2f}]"
    else:
        ad_annotation += "<br>95% CI: [nan, nan]"
    
    fig.add_annotation(
        x=0.75,   # Position outside the plot
        y=-0.22,   # Position above the plot
        xref="paper",
        yref="paper",
        text=ad_annotation,
        showarrow=False,
        font=dict(size=12, color="red"),
        align="left",
        bordercolor="red",
        borderwidth=1,
        borderpad=4,
        xanchor="center",
        yanchor="top"
    )
    
    # Update layout with more margin at the top for annotations
    fig.update_layout(
        height=500,
        margin=dict(t=80,b=200),  # Increased top margin
        showlegend=True,
        title_text="Spearman Correlation Analysis with Pro-inflammatory Lipid Index"
        )
    
    # Update axes
    for i in [1, 2]:
        fig.update_xaxes(
            title_text="Pro-inflammatory Lipid Index",
            row=1, col=i
        )
        fig.update_yaxes(
            title_text="Disease Prevalence (%)",
            row=1, col=i
        )
    
    return fig

def main():
    """
    Main execution function that runs the complete analysis pipeline.
    """
    st.title("Pro-inflammatory Lipid Index Analysis _ Luan Vu UTSA and Joan Cook-Mills IU_ Dec 2024")
    
    # Add methodology explanation
    st.markdown("""
    ### Methodology and Statistical Choices
    
    **1. Pro-inflammatory Index Calculation:**
    - Based on log2 fold changes relative to control cluster (C6)
    - Incorporates both pro- and anti-inflammatory markers
    - α-tocopherol contribution is inversed due to its protective effects
    - γ-tocopherol contribution remains positive due to pro-inflammatory effects
    
    **2. Statistical Analysis:**
    - Spearman correlation coefficient (ρ) used due to:
        * Small sample size (n=7 clusters)
        * No assumption of normal distribution
        * Robust to outliers
    - Bootstrap confidence intervals (5000 resamples) for robust uncertainty estimates
    """)
    
    # Load data
    df = load_and_preprocess_data('data.csv')
    
    # Stage 1
    st.header("Stage 1: Initial Data Analysis")

    # Display missing data analysis
    st.subheader("Missing Data Analysis")
    missing_data = analyze_missing_data(df)
    st.dataframe(missing_data)

    # Add user choice for handling missing data
    handle_missing = st.radio(
        "How would you like to handle missing data?",
        options=['Keep all data', 'Remove rows with missing values'],
        index=0  # Default to keeping all data
    )

    # Create a copy of the dataframe for analysis
    analysis_df = df.copy()
    if handle_missing == 'Remove rows with missing values':
        # Only remove rows where tocopherol values are missing
        df = df.dropna()
        st.write(f"Number of samples after removing missing values: {len(df)}")

    tocopherol_stats = analyze_tocopherol_distribution(df)
    stage1_plots = create_stage1_plots(df)
    
    st.plotly_chart(stage1_plots['distributions'])
    st.plotly_chart(stage1_plots['outcomes'])
    
    # Stage 2
    st.header("Stage 2: Statistical Analysis")
    stats_results = {
        'alpha': compare_with_control(df, 'alpha-tocopherol'),
        'gamma': compare_with_control(df, 'gamma-tocopherol')
    }
    stage2_plot = create_stage2_plots(df, stats_results)
    st.plotly_chart(stage2_plot)
    
    # Stage 3
    st.header("Stage 3: Pro-inflammatory Lipid Index")
    proinflammatory_df = calculate_proinflammatory_index(df)
    
    # Display the pro-inflammatory index table
    st.subheader("Pro-inflammatory Index Results")
    
    # Format the dataframe properly
    formatted_df = proinflammatory_df.copy()
    numeric_columns = [
        'βGlcCers (mean log2FC)',
        'βGalCers (mean log2FC)', 
        'sphingomyelins (mean log2FC)',
        'Lipoxin A4 (log2FC)',
        'Resolvin D2 (log2FC)',
        'alpha-tocopherol (log2FC)',
        'gamma-tocopherol (log2FC)',
        'pro-inflammatory lipid index'
    ]
    
    # Apply formatting while preserving None values
    for col in numeric_columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f'{x:.2f}' if pd.notnull(x) else None
        )
    
    # Display the formatted dataframe
    st.dataframe(formatted_df)
    
    # Correlation analysis
    correlation_results = perform_correlation_analysis(df, proinflammatory_df)
    correlation_plot = create_correlation_plots(df, proinflammatory_df, correlation_results)
    
    st.plotly_chart(correlation_plot)

if __name__ == "__main__":
    main()