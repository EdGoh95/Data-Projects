#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 15 - Quality and acceptance testing
"""
import shap
import matplotlib.pyplot as plt
from collections import namedtuple

class ImageHandling:
    def __init__(self, fig, name):
        self.fig = fig
        self.name = name

    def _resize_plot(self):
        self.fig = plt.gcf()
        self.fig.set_size_inches(12, 12)

    def save_base(self):
        self.fig.savefig('{}.png'.format(self.name), bbox_inches = 'tight', dpi = 600)
        self.fig.savefig('{}.svg'.format(self.name), bbox_inches = 'tight', dpi = 600)

    def save_plot(self):
        self._resize_plot()
        self.save_base()

    def save_js(self):
        shap.save_html(self.name, self.fig)
        return self.fig

class ShapConstructor:
    def __init__(self, base_values, data, values, feature_names, shape):
        self.base_values = base_values
        self.data = data
        self.values = values
        self.feature_names = feature_names
        self.shape = shape

class ShapObject:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.explainer = self.generate_explainer(self.model, self.data)
        self.initjs()

    @classmethod
    def generate_explainer(self, model, data):
        Explain = namedtuple('Explain', 'shap_values, explainer, max_rows')
        explainer = shap.Explainer(model)
        explainer.expected_value = explainer.expected_value[0]
        shap_values = explainer(data)
        max_rows = len(shap_values.values)
        return Explain(shap_values, explainer, max_rows)

    def build(self, row = 0):
        return ShapConstructor(
            base_values = self.explainer.shap_values[0][0].base_values,
            values = self.explainer.shap_values[row].values, feature_names = self.data.columns,
            data = self.explainer.shap_values[0].data, shape = self.explainer.shap_values[0].shape)

    def validate_row(self, row):
        assert row < self.explainer.max_rows, 'Row number {} is invalid. Data has only {} rows.'.format(
            row, self.explainer.max_rows)

    def plot_waterfall(self, row = 0):
        plt.clf()
        self.validate_row(row)
        fig = shap.waterfall_plot(self.build(row), show = False, max_display = 15)
        ImageHandling(fig, '{} Summary Plot'.format(row)).save_plot()
        return fig

    def plot_summary(self):
        fig = shap.plots.beeswarm(self.explainer.shap_values, show = False, max_display = 15)
        ImageHandling(fig, 'Summary Plot').save_plot()

    def plot_force_by_row(self, row = 0):
        plt.clf()
        self.validate_row(row)
        fig = shap.force_plot(self.explainer.explainer.expected_value,
                              self.explainer.shap_values.values[row, :], self.data.iloc[row, :],
                              show = False, matplotlib = True)
        ImageHandling(fig, '{} Force Plot'.format(row)).save_base()

    def plot_force_full(self):
        fig = shap.plots.force(self.explainer.explainer.expected_value, self.explainer.shap_values.values,
                               show = False)
        return ImageHandling(fig, 'Force Plot (Full).html').save_js()

    def plot_importance(self):
        fig = shap.plots.bar(self.explainer.shap_values, show = False, max_display = 15)
        ImageHandling(fig, 'SHAP Importance Plot').save_plot()

    def plot_scatter(self, feature):
        fig = shap.plots.scatter(self.explainer.shap_values[:, feature], color = self.explainer.shap_values,
                                 show = False)
        ImageHandling(fig, '{} Scatter Plot'.format(feature)).save_plot()