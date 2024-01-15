#!/usr/bin/env python3
import warnings
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import matplotlib.pyplot as plt
from collections import namedtuple
from statsmodels.graphics.gofplots import qqplot
from statsmodels.formula.api import ols
from statsmodels.stats import anova, multicomp

class DistributionAnalysis(object):
    def __init__(self, series, histogram_bins, **kwargs):
        self.series = series
        self.histogram_bins = histogram_bins
        self.series_name = kwargs.get('series_name', 'data')
        self.plot_bins = kwargs.get('plot_bins', 200)
        self.best_plot_size = kwargs.get('best_plot_size', (15, 12))
        self.all_plot_size = kwargs.get('all_plot_size', (20, 25))
        self.MIN_BOUNDARY = 0.001
        self.MAX_BOUNDARY = 0.999
        self.ALPHA = kwargs.get('alpha', 0.05)

    def _get_series_bins(self):
        return int(np.ceil(self.series.index.values.max()))

    @staticmethod
    def _get_distributions():
        scipy_version = scipy.__version__
        if (int(scipy_version[2]) >= 5) and (int(scipy_version[4:]) > 3):
            names, gen_names = stat.get_distribution_names(stat.pairs, stat.rv_continuous)
        else:
            names = stat._continuous_distns._distn_names
        return names

    @staticmethod
    def _extract_params(params):
        return {'arguments': params[:-2], 'location': params[-2], 'scale': params[-1]}

    @staticmethod
    def _generate_boundaries(distribution, parameters, x):
        return distribution.ppf(
            x, *parameters['arguments'], loc = parameters['location'],
            scale = parameters['scale']) if parameters['arguments'] else distribution.ppf(
                x, loc = parameters['location'], scale = parameters['scale'])

    @staticmethod
    def _build_pdf(x, distribution, parameters):
        if parameters['arguments']:
            pdf = distribution.pdf(x, loc = parameters['location'], scale = parameters['scale'],
                                   *parameters['arguments'])
        else:
            pdf = distribution.pdf(x, loc = parameters['location'], scale = parameters['scale'])
        return pdf

    def plot_normalcy(self):
        qqplot(self.series, line = 's')

    def check_normalcy(self):
        def significance_test(value, threshold):
            return 'Data set {} normally distributed from'.format('is' if value > threshold
                                                                   else 'is not')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            shapiro_stat, shapiro_p_value = stat.shapiro(self.series)
            dagostino_stat, dagostino_p_value = stat.normaltest(self.series)
            anderson_stat, anderson_critical_values, anderson_significance_levels = stat.anderson(self.series)
            anderson_report = list(zip(list(anderson_critical_values), list(anderson_significance_levels)))
            shapiro_statement = ('{} Shapiro-Wilt Normality Test with statistic = {:.4f} and '
            'p-value = {:.4e}'.format(significance_test(shapiro_p_value, self.ALPHA), shapiro_stat,
            shapiro_p_value))
            dagostino_statement = ("\n{} D'Agostino & Pearson's Test with statistic = {:.4f} and "
            'p-value = {:.4e}'.format(significance_test(dagostino_p_value, self.ALPHA), dagostino_stat,
            dagostino_p_value))
            anderson_statement = '\nAnderson Statistics = {:.4f}'.format(anderson_stat)
            for a in anderson_report:
                anderson_statement = anderson_statement + ('\nFor significance level {} of '
                'Anderson-Darling Test: {} the evaluation. Critical value: {}'.format(
                a[1], significance_test(a[0], anderson_stat), a[0]))
        return '{}{}{}'.format(shapiro_statement, dagostino_statement, anderson_statement)

    def _calculate_fit_loss(self, x, y, dist):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            estimated_distribution = dist.fit(x)
            pdf = self._build_pdf(x, dist, self._extract_params(estimated_distribution))
        return np.sum(np.power(y - pdf, 2.0)), estimated_distribution

    def _generate_probability_distribution(self, distribution, parameters, bins):
        start_point = self._generate_boundaries(distribution, parameters, self.MIN_BOUNDARY)
        end_point = self._generate_boundaries(distribution, parameters, self.MAX_BOUNDARY)
        x = np.linspace(start_point, end_point, bins)
        y = self._build_pdf(x, distribution, parameters)
        return pd.Series(y, x)

    def find_distribution_fit(self):
        y_histogram, x_histogram_raw = np.histogram(self.series, self.histogram_bins, density = True)
        x_histogram = (x_histogram_raw + np.roll(x_histogram_raw, -1))[:-1]/2
        full_distribution_results = {}
        best_loss = np.inf
        best_fit = stat.norm
        best_parameters = (0.0, 1.0)
        for distribution in self._get_distributions():
            histogram = getattr(stat, distribution)
            results, parameters = self._calculate_fit_loss(x_histogram, y_histogram, histogram)
            full_distribution_results[distribution] = {
                'histogram': histogram, 'loss': results, 'parameters': {
                    'arguments': parameters[:-2], 'location': parameters[-2], 'scale': parameters[-1]}}
            if best_loss > results > 0:
                best_loss = results
                best_fit = histogram
                best_parameters = parameters
        return {'best_distribution': best_fit, 'best_loss': best_loss, 'best_parameters': {
            'arguments': best_parameters[:-2], 'location': best_parameters[-2],
            'scale': best_parameters[-1]}, 'all_results': full_distribution_results}

    def plot_best_fit(self):
        fits = self.find_distribution_fit()
        best_fit_distribution = fits['best_distribution']
        best_fit_parameters = fits['best_parameters']
        distribution_series = self._generate_probability_distribution(
            best_fit_distribution, best_fit_parameters, self._get_series_bins())
        with plt.style.context(style = 'seaborn-v0_8'):
            fig = plt.figure(figsize = self.best_plot_size)
            ax = self.series.plot(kind = 'hist', bins = self.plot_bins, density = True, alpha = 0.5,
                                  label = self.series_name, legend = True)
            distribution_series.plot(lw = 3, label = best_fit_distribution.__class__.__name__,
                                     legend = True, ax = ax)
            ax.legend(loc = 'best')
        return fig

    def plot_all_fits(self):
        fits = self.find_distribution_fit()
        series_bins = self._get_series_bins()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with plt.style.context(style = 'seaborn-v0_8'):
                fig = plt.figure(figsize = self.all_plot_size)
                ax = self.series.plot(kind = 'hist', bins = self.plot_bins, density = True,
                                      alpha = 0.5, label = self.series_name, legend = True)
                x_max = ax.get_xlim()
                y_max = ax.get_ylim()
                for distribution in fits['all_results']:
                    histogram = fits['all_results'][distribution]
                    distribution_data = self._generate_probability_distribution(
                        histogram['histogram'], histogram['parameters'], series_bins)
                    distribution_data.plot(lw = 2, label = distribution, alpha = 0.5, ax = ax)
                ax.legend(loc = 'best')
                ax.set_xlim(x_max)
                ax.set_ylim(y_max)
        return fig

#%% Unit Tests
def test_generate_boundaries():
    expected_lower_norm = -2.3263478740408408
    expected_upper_norm = 2.3263478740408408
    boundary_arguments = {'location': 0, 'scale': 1, 'arguments': {}}
    test_object = DistributionAnalysis(np.arange(0, 100), 10)
    normal_distribution_lower = test_object._generate_boundaries(stat.norm, boundary_arguments, 0.01)
    normal_distribution_upper = test_object._generate_boundaries(stat.norm, boundary_arguments, 0.99)
    assert normal_distribution_lower == expected_lower_norm, '''Lower boundary of the Normal
    Distribution: {} does not match expected: {}'''.format(normal_distribution_lower, expected_lower_norm)
    assert normal_distribution_upper == expected_upper_norm, '''Upper boundary of the Normal
    Distribution: {} does not match expected: {}'''.format(normal_distribution_upper, expected_upper_norm)

#%% Statistical Testing
def calculate_basic_stats_df(series):
    StatsData = namedtuple('StatsData', 'name, mean, median, stddev, variance, sum')
    return StatsData(series.name, np.mean(series), np.median(series), np.std(series), np.var(series),
                     np.sum(series))
def series_comparison_continuous_df(a, b):
    BatteryData = namedtuple('BatteryData', 'left, right, anova, mann_whitney_u, wilcoxon, t_test')
    TestData = namedtuple('TestData', 'statistic, pvalue')
    anova_test = stat.f_oneway(a, b)
    mann_whitney_test = stat.mannwhitneyu(a, b)
    wilcoxon_rank_test = stat.wilcoxon(a, b)
    t_test = stat.ttest_ind(a, b, equal_var = False)
    return BatteryData(a.name, b.name, TestData(anova_test.statistic, anova_test.pvalue),
                       TestData(mann_whitney_test.statistic, mann_whitney_test.pvalue),
                       TestData(wilcoxon_rank_test.statistic, wilcoxon_rank_test.pvalue),
                       TestData(t_test.statistic, t_test.pvalue))

def plot_comparison_series_df(x, y1, y2, name, plot_size = (10, 10)):
    with plt.style.context(style = 'seaborn-v0_8'):
        fig = plt.figure(figsize = plot_size)
        ax = fig.add_subplot(111)
        ax.plot(x, y1, color = 'red', label = y1.name)
        ax.plot(x, y2, color = 'green', label = y2.name)
        ax.set_title('Comparison Of Sales Betweem Tests {} and {}'.format(y1.name, y2.name))
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')

        comparison = series_comparison_continuous_df(y1, y2)
        y1_stats = calculate_basic_stats_df(y1)
        y2_stats = calculate_basic_stats_df(y2)
        bbox_stats = '\n'.join(('Series {}'.format(y1.name),
                                ' - Mean: {:.2f}'.format(y1_stats.mean),
                                ' - Median: {:.2f}'.format(y1_stats.median),
                                ' - Standard Deviation: {:.2f}'.format(y1_stats.stddev),
                                ' - Variance: {:.2f}'.format(y1_stats.variance),
                                ' - Sum: {:.2f}'.format(y1_stats.sum),
                                '\nSeries {}'.format(y2.name),
                                ' - Mean: {:.2f}'.format(y2_stats.mean),
                                ' - Median: {:.2f}'.format(y2_stats.median),
                                ' - Standard Deviation: {:.2f}'.format(y2_stats.stddev),
                                ' - Variance: {:.2f}'.format(y2_stats.variance),
                                ' - Sum: {:.2f}'.format(y2_stats.sum)))
        bbox_text = '\n'.join(('ANOVA Test p-value: {}'.format(comparison.anova.pvalue),
                              't-Test p-value: {}'.format(comparison.t_test.pvalue),
                              'Mann-Whitney U Test p-value: {}'.format(comparison.mann_whitney_u.pvalue),
                              'Wilcoxon Signed-Rank Test p-value: {}'.format(comparison.wilcoxon.pvalue)))
        bbox_props = dict(boxstyle = 'round', facecolor = 'ivory', alpha = 0.8)
        ax.text(0.05, 0.95, bbox_text, transform = ax.transAxes, fontsize = 12,
                verticalalignment = 'top', bbox = bbox_props)
        ax.text(0.05, 0.8, bbox_stats, transform = ax.transAxes, fontsize = 10,
                verticalalignment = 'top', bbox = bbox_props)
        ax.legend(loc = 'lower right')
        plt.tight_layout()
        plt.savefig('Time Series - {}.jpg'.format(name), dpi = 600)

# Evaluation of continuous metrics
def run_anova(data, value_name, group_name):
    ols_model = ols('{} ~ C({})'.format(value_name, group_name), data = data).fit()
    return anova.anova_lm(ols_model, typ = 2)

def plot_anova(melted_df, plot_name, plot_size = (16, 16)):
    anova_report = run_anova(melted_df, 'Sales', 'Test')
    with plt.style.context(style = 'seaborn-v0_8'):
        fig = plt.figure(figsize = plot_size)
        fig.add_subplot(111)
        sns.boxplot(x = 'Test', y = 'Sales', data = melted_df, color = 'lightsteelblue')
        sns.stripplot(x = 'Test', y = 'Sales', data = melted_df, color = 'steelblue',
                            alpha = 0.4, jitter = 0.2)

        ax1 = fig.add_subplot(211)
        ax1.set_title('ANOVA Analysis', y = 1.2, fontsize = 20)
        table = ax1.table(cellText = anova_report.values, colLabels = anova_report.columns,
                          rowLabels = anova_report.index, loc = 'top', cellLoc = 'center',
                          rowLoc = 'center', bbox = [0.075, 1.0, 0.875, 0.2])
        table.auto_set_column_width(col = list(range(len(anova_report.columns))))
        ax1.axis('tight')
        ax1.set_axis_off()
        plt.tight_layout()
        plt.savefig('ANOVA Analysis - {}.jpg'.format(plot_name), dpi = 600)

def extract_data_from_table(result_data, index, data_type):
    return pd.Series([b[index] for b in result_data], dtype = np.dtype(data_type))

def convert_tukey_results_to_df(tukey_results_table):
    data_types =  [(0, 'str'), (1, 'str'), (2, 'float'), (3, 'float'), (4, 'float'), (5, 'float'),
                   (6, 'bool')]
    extracted_data = [extract_data_from_table(tukey_results_table.data[1:], d[0], d[1]) for d in data_types]
    results_df = pd.concat(extracted_data, axis = 1)
    results_df.columns = tukey_results_table.data[0]
    return results_df.sort_values(['p-adj', 'meandiff'], ascending = [True, False])

def run_tukey(value, group, alpha = 0.05):
    paired_test = multicomp.pairwise_tukeyhsd(value, group, alpha)
    return convert_tukey_results_to_df(paired_test._results_table)

def plot_tukey(melted_df, plot_name, plot_size = (14, 14)):
    tukey_report = run_tukey(melted_df['Sales'], melted_df['Test'])
    with plt.style.context(style = 'seaborn-v0_8'):
        fig = plt.figure(figsize = plot_size)
        fig.add_subplot(111)
        sns.boxplot(x = 'Test', y = 'Sales', data = melted_df, color = 'lightsteelblue')
        sns.stripplot(x = 'Test', y = 'Sales', data = melted_df, color = 'steelblue',
                      alpha = 0.4, jitter = 0.2)

        ax_table = fig.add_subplot(211)
        ax_table.set_title('Tukey Honestly Significant Difference (HSD) Test Analysis', y = 1.5,
                           fontsize = 20)
        table = ax_table.table(cellText = tukey_report.values, colLabels = tukey_report.columns,
                               rowLabels = tukey_report.index, loc = 'top', cellLoc = 'center',
                               rowLoc = 'center', bbox = [0.075, 1.0, 0.875, 0.5])
        table.auto_set_column_width(col = list(range(len(tukey_report.columns))))
        ax_table.axis('tight')
        ax_table.set_axis_off()
        plt.tight_layout()
        plt.savefig('Tukey HSD Test Analysis - {}.svg'.format(plot_name), dpi = 600)

# Evaluation of categorical metrics
def categorical_significance(test_happen, test_not_happen, control_happen, control_not_happen):
    CategoricalTest = namedtuple(
        'CategoricalTest', 'fisher_stat, fisher_pvalue, chisq_stat, chisq_pvalue, chisq_df, chisq_expected')
    test_happen_total = np.sum(test_happen)
    test_not_happen_total = np.sum(test_not_happen)
    control_happen_total = np.sum(control_happen)
    control_not_happen_total = np.sum(control_not_happen)
    matrix_total = np.array([[test_happen_total, control_happen_total],
                             [test_not_happen_total, control_not_happen_total]])
    fisher_stat, fisher_pvalue = stat.fisher_exact(matrix_total)
    chisq_stat, chisq_pvalue, chisq_df, chisq_expected = stat.chi2_contingency(matrix_total)
    return CategoricalTest(fisher_stat, fisher_pvalue, chisq_stat, chisq_pvalue, chisq_df, chisq_expected)

def plot_coupon_usage(test_happen, test_not_happen, control_happen, control_not_happen, date_range,
                      name, plot_size = (10, 8)):
    categorical_test = categorical_significance(test_happen, test_not_happen, control_happen,
                                                control_not_happen)
    with plt.style.context(style = 'seaborn-v0_8'):
        fig = plt.figure(figsize = plot_size)
        ax = fig.add_subplot(111)
        ax.bar(date_range, test_happen, color = '#5499C7', label = 'Coupons Used (Test Group)')
        ax.bar(date_range, test_not_happen, bottom = test_happen, color = '#A9CCE3',
               label = 'Coupons Unused (Test Group)')
        ax.bar(date_range, control_happen, bottom = test_happen + test_not_happen, color = '#52BE80',
               label = 'Coupons Used (Control Group)')
        ax.bar(date_range, control_not_happen, bottom = test_happen + test_not_happen + control_happen,
               color = '#A9DFBF', label = 'Coupons Unused (Control Group)')
        bbox_text = '\n'.join(('Fisher p-value: {:.2e}'.format(categorical_test.fisher_pvalue),
                               r'$\chi^2$ Contingency p-value: {:.2e}'.format(categorical_test.chisq_pvalue),
                               r'$\chi^2$ Degrees of Freedom: {}'.format(categorical_test.chisq_df)))
        bbox_props = dict(boxstyle = 'round', facecolor = 'ivory', alpha = 1.0)
        ax.set_title ('Comparison of Coupon Usage Between The Control And Test Groups', fontsize = 16)
        ax.text(0.05, 0.95, bbox_text, transform = ax.transAxes, fontsize = 12,
                verticalalignment = 'top', bbox = bbox_props)
        ax.set_xlabel('Date')
        ax.set_ylabel('Coupon Usage')
        legend = ax.legend(loc = 'best', shadow = True, frameon = True)
        legend.get_frame().set_facecolor('ivory')
        plt.tight_layout()
        plt.savefig('Coupon Usage Over {}.svg'.format(name), dpi = 600)

if __name__ == '__main__':
    test_generate_boundaries()
    print('Test Results: Passed!')
    airbnb_df = pd.read_parquet('sf-listings-2019-03-06-clean.parquet')
    DistributionAnalysis(series = airbnb_df['price'], histogram_bins = 200,
                         series_name = 'Airbnb SF Prices').plot_best_fit()