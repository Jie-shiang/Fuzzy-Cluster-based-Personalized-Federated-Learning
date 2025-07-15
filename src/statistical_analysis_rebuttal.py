import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_data():
    # TODO: Load CSV with both FCPFL and baseline WER for each utterance
    # Expected CSV format:
    # utterance_id, fcpfl_wer, baseline_wer, text, other_features...
    
    # Placeholder - replace with actual CSV loading
    df = pd.DataFrame({
        'utterance_id': [],
        'fcpfl_wer': [],
        'baseline_wer': [],
        'text': []
    })
    
    return df

def true_paired_ttest_analysis(df):
    print("=== True Paired t-test Analysis ===")
    
    # Extract paired WER data
    fcpfl_wers = df['fcpfl_wer'].values
    baseline_wers = df['baseline_wer'].values
    
    print(f"Number of utterance pairs: {len(fcpfl_wers)}")
    print(f"FCPFL mean WER: {np.mean(fcpfl_wers):.4f}")
    print(f"Baseline mean WER: {np.mean(baseline_wers):.4f}")
    
    # Perform true paired t-test
    # H0: no difference between paired samples
    # H1: FCPFL < Baseline (one-tailed test for improvement)
    t_statistic, p_value_two_tailed = stats.ttest_rel(baseline_wers, fcpfl_wers)
    p_value_one_tailed = p_value_two_tailed / 2
    
    print(f"\nPaired t-test Results:")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value (two-tailed): {p_value_two_tailed:.6f}")
    print(f"p-value (one-tailed): {p_value_one_tailed:.6f}")
    print(f"Degrees of freedom: {len(fcpfl_wers)-1}")
    
    # Statistical significance
    alpha = 0.05
    significant = "Yes" if p_value_one_tailed < alpha else "No"
    print(f"Statistically significant (p < {alpha}): {significant}")
    
    # Effect size (Cohen's d for paired samples)
    differences = baseline_wers - fcpfl_wers
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    print(f"Cohen's d (effect size): {cohens_d:.4f}")
    
    def interpret_effect_size(d):
        if abs(d) < 0.2:
            return "Small"
        elif abs(d) < 0.5:
            return "Medium" 
        elif abs(d) < 0.8:
            return "Large"
        else:
            return "Very Large"
    
    print(f"Effect size interpretation: {interpret_effect_size(cohens_d)}")
    
    # 95% Confidence interval for the mean difference
    confidence_level = 0.95
    degrees_freedom = len(differences) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_critical * (np.std(differences, ddof=1) / np.sqrt(len(differences)))
    
    ci_lower = np.mean(differences) - margin_error
    ci_upper = np.mean(differences) + margin_error
    
    print(f"95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Relative improvement
    relative_improvement = (np.mean(differences) / np.mean(baseline_wers)) * 100
    print(f"Relative improvement: {relative_improvement:.2f}%")
    
    return {
        'fcpfl_wers': fcpfl_wers,
        'baseline_wers': baseline_wers,
        'differences': differences,
        't_statistic': t_statistic,
        'p_value': p_value_one_tailed,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'relative_improvement': relative_improvement
    }

def bootstrap_paired_analysis(df):
    print("\n=== Bootstrap Paired Analysis ===")
    
    fcpfl_wers = df['fcpfl_wer'].values
    baseline_wers = df['baseline_wer'].values
    observed_differences = baseline_wers - fcpfl_wers
    
    n_bootstrap = 1000
    bootstrap_means = []
    
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        # Bootstrap resampling of paired differences
        bootstrap_indices = np.random.choice(len(observed_differences), 
                                           size=len(observed_differences), 
                                           replace=True)
        bootstrap_differences = observed_differences[bootstrap_indices]
        bootstrap_means.append(np.mean(bootstrap_differences))
    
    bootstrap_means = np.array(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    print(f"Bootstrap results ({n_bootstrap} iterations):")
    print(f"Mean improvement: {np.mean(bootstrap_means):.4f}")
    print(f"Standard error: {np.std(bootstrap_means):.4f}")
    print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    significant = "Yes" if ci_lower > 0 else "No"
    print(f"CI excludes zero (significant): {significant}")
    
    return bootstrap_means, ci_lower, ci_upper

def validate_paired_assumptions(df):
    print("\n=== Paired t-test Assumptions Check ===")
    
    differences = df['baseline_wer'].values - df['fcpfl_wer'].values
    
    # Normality test (Shapiro-Wilk)
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    print(f"Shapiro-Wilk normality test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  p-value: {shapiro_p:.4f}")
    normal = "Yes" if shapiro_p > 0.05 else "No"
    print(f"  Differences normally distributed: {normal}")
    
    # Basic descriptive statistics of differences
    print(f"\nDifferences descriptive statistics:")
    print(f"  Mean: {np.mean(differences):.4f}")
    print(f"  Std: {np.std(differences, ddof=1):.4f}")
    print(f"  Min: {np.min(differences):.4f}")
    print(f"  Max: {np.max(differences):.4f}")

def generate_summary(df, comparison_results, bootstrap_results):
    print("\n" + "="*50)
    print("PAIRED T-TEST STATISTICAL SUMMARY")
    print("="*50)
    
    print(f"Dataset: {len(df)} paired utterances")
    
    print(f"\nStatistical Significance:")
    print(f"- Paired t-test p-value: {comparison_results['p_value']:.6f}")
    print(f"- Statistically significant: {'Yes' if comparison_results['p_value'] < 0.05 else 'No'}")
    print(f"- Cohen's d: {comparison_results['cohens_d']:.4f}")
    
    bootstrap_means, ci_lower, ci_upper = bootstrap_results
    print(f"\nConfidence Intervals:")
    print(f"- 95% t-test CI: [{comparison_results['ci_lower']:.4f}, {comparison_results['ci_upper']:.4f}]")
    print(f"- 95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nImprovement:")
    print(f"- Relative improvement: {comparison_results['relative_improvement']:.2f}%")
    print(f"- Mean absolute improvement: {np.mean(comparison_results['differences']):.4f}")

def main():
    print("True Paired t-test Statistical Analysis")
    print("="*40)
    
    df = load_data()
    
    if len(df) == 0:
        print("ERROR: No data loaded. Please provide CSV with paired WER results.")
        print("Expected columns: utterance_id, fcpfl_wer, baseline_wer")
        return
    
    # Validate assumptions
    validate_paired_assumptions(df)
    
    # Perform paired analysis
    comparison_results = true_paired_ttest_analysis(df)
    bootstrap_results = bootstrap_paired_analysis(df)
    
    # Generate summary
    generate_summary(df, comparison_results, bootstrap_results)

if __name__ == "__main__":
    main()