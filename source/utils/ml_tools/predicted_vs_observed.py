import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    recall_score,
    precision_score
)

from source.utils.model_output import ModelOutput


class PvO:

    def __init__(self, predicted, observed):
        self.predicted = predicted
        self.observed = observed


class PredictedAndObserved(ModelOutput):
    """
    Base class for PredictedAndObservedClassification and PredictedAndObservedRegression.

    """

    def __init__(self, predicted, observed, quantiles, output_dir=None, index=None, group=None):
        super().__init__(output_dir=output_dir)
        self.predicted = pd.Series(predicted)
        self.observed = pd.Series(observed.values)
        self.quantiles = quantiles
        self.quantiles_bins = np.arange(0, 1 + 1 / self.quantiles, 1 / self.quantiles)
        self.predicted_percent_rank = self.predicted.rank(pct=True)
        self.quantile = pd.cut(x=self.predicted_percent_rank, bins=self.quantiles_bins)
        self.quantile_mid = self.quantile.apply(lambda x: x.mid)
        self.error = (self.predicted - self.observed).abs()
        self.output_dir = output_dir
        self.quantile_mean_prediction = self.predicted.groupby(self.quantile_mid).mean()
        self.quantile_mean_observed = self.observed.groupby(self.quantile_mid).mean()
        self.index = index
        # self.quantile_index = [el for el in [self.quantile_mid, group] if el is not None]
        self.group = group

        # prepare prediction data frame
        self.individual_predicted_and_observed = (
            pd.DataFrame(data={
                'predicted': self.predicted,
                "observed": self.observed
            }
            )
                .assign(predicted_percent_rank=self.predicted_percent_rank)
                .assign(quantile=self.quantile)
                .assign(quantile_mid=self.quantile_mid)
                .assign(error=self.error)
                .set_index(self.index)  # set index after everything else
        )

    def get_pvo(self):
        """

        Predicted vs. Observed (PvO) charts are one of the first tests you should perform on a fitted model.
        They help you to understand:
        - Whether there is any bias at different prediction levels
        - The range and distribution of predictions
        - Whether the model has been overfit (but comparing PvO charts created using the train and test_2 /
        holdout datasets)

        The chart is created by plotting the average predicted_validation and observed_validation value for each predicted_validation percentile
        (i.e. percentiles of the predicted_validation values)

        .. plot::

            from source.pyplots.gains import pvo_classification
            pvo_classification.plot_pvo()

        """
        observed = (
            self
                .observed
                .groupby(self.quantile_mid)
                .mean()
        )
        predicted = (
            self
                .predicted
                .groupby(self.quantile_mid)
                .mean()
        )

        return PvO(predicted, observed)

    def get_individual_predicted_and_observed(self):
        return self.individual_predicted_and_observed

    def plot_pvo(self):
        """

        Plots predicted_validation and observed_validation by quantile

        see :meth:`PredictedAndObserved.get_pvo`

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.lineplot(
            x=self.get_pvo().predicted.index,
            y=self.get_pvo().predicted,
            ax=ax,
            label='predicted'
        )
        sns.lineplot(
            x=self.get_pvo().observed.index,
            y=self.get_pvo().observed,
            ax=ax,
            label='observed'
        )

        ax.set_xlabel('quantile')
        ax.set_ylabel('average prediction and observation by quantile')
        ax.set_title("predicted vs observed")
        ax.legend()

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename=f'pvo_{self.quantiles}'
        )

    def get_pvo_error(self):
        """

        Predicted vs. observed_validation error is defined as predicted_validation probability - observed_validation value. Can be used as a diagnostic
        to determine any bias at different prediction levels

        .. plot::

            from source.pyplots.gains import pvo_classification
            pvo_classification.plot_pvo_quantile_error()


        """
        pvo = self.get_pvo()

        pvo_error = pvo.predicted - pvo.observed

        return pvo_error

    def get_prediction_error(self):
        return (
                self.predicted - self.observed
        )

    def plot_pvo_quantile_error(self):
        """

        Plots predicted_validation vs observed_validation error

        see :meth:`PredictedAndObserved.get_pvo_error`

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.barplot(
            x=self.get_pvo_error().index,
            y=self.get_pvo_error(),
            color="black",
            ax=ax
        )

        ax.set_xlabel('quantile')
        ax.set_ylabel('error: predicted- observed')
        ax.set_title("PvO error")

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename=f'lift_{self.quantiles}'
        )

    def plot_prediction_error(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.distplot(
            a=self.get_prediction_error(),
            bins=self.quantiles,
            color="black",
            ax=ax
        )

        ax.set_xlabel('error')
        ax.set_ylabel('density')
        ax.set_title("Distribution of prediction errors")

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename='pvo_error_{self.quantiles}'
        )

    def get_metrics(self) -> dict:
        raise NotImplementedError("Subclass must implement")

    def get_uplift_from_baseline(self, metric: str) -> dict:
        return {
            'baseline': self.get_metrics().get(metric),
            'uplift_model': self.get_metrics().get('baseline').get('metric'),
            'absolute_uplift': self.get_metrics().get(metric) - self.get_metrics().get('baseline').get('metric'),
            'relative_uplift': self.get_metrics().get(metric) / self.get_metrics().get('baseline').get('metric')
        }


# class GroupPredictedAndObserved(PredictedAndObserved):
#     def __init__(self, group: List[str], grouped_predicted_and_observed: List[PredictedAndObserved]):
#         self.group = group
#         self.predicted_and_observed = grouped_predicted_and_observed
#         self.group_with_predicted_and_observed = dict(zip(self.group, self.predicted_and_observed))
#         super(PredictedAndObserved, self)._init__(
#             predicted=predicted,
#             observed=observed,
#             quantiles=quantiles,
#             output_dir=output_dir
#         )
#
#     def get_group_predicted_and_oberved(self, group):
#         return self.group_with_predicted_and_observed.get(group, None)
#
#
#     def make_group_predicted_and_observed(self):
#         return (
#             GroupPredictedAndObserved(
#
#             )
#         )


class PredictedAndObservedClassification(PredictedAndObserved):
    """
    PredictedAndObservedClassification provides an interface to interact with several diagnostic, including plots relating to
    predicted_validation probabilities and observed_validation values.

    The interface includes:
        - lift
        - cumulative lift
        - gains
        - predicted_validation vs. observed_validation (pvo_validation)
        - prediction errors
        - gini calculation

    and where appropriate plots for the above figures


    """

    # def create_individual_prediction_statistics(
    #         self,
    #         quantile: int = 100
    # ) -> pd.DataFrame:
    #     return (
    #         pd.DataFrame({
    #             'predicted': self.predicted,
    #             'observed': self.observed
    #         })
    #             .assign(probability_zero=lambda correlations: 1 - correlations.predicted)
    #             .assign(observed=lambda correlations: correlations.observed)
    #             .assign(prediction_error=lambda correlations: np.abs(correlations.predicted - correlations.observed))
    #             .assign(percent_rank_probablity_one=lambda correlations: calc_percent_rank(correlations.predicted))
    #             .assign(
    #             percentile_one=lambda correlations: self.calc_quantile_from_percent_rank(correlations.percent_rank_probablity_one,
    #                                                                           quantiles=quantile))
    #             .assign(true_positive=lambda correlations: self.calc_true_positive(correlations.predicted_class, correlations.observed))
    #             .assign(false_positive=lambda correlations: self.calc_false_positive(correlations.predicted_class, correlations.observed))
    #             .assign(true_negative=lambda correlations: self.calc_true_negative(correlations.predicted_class, correlations.observed))
    #             .assign(false_negative=lambda correlations: self.calc_false_negative(correlations.predicted_class, correlations.observed))
    #     )
    #
    # @staticmethod
    # def calc_quantile_from_percent_rank(percent_ranked: pd.Series, quantiles: int = 100) -> pd.Series:
    #     return pd.cut(
    #         correlations=percent_ranked,
    #         bins=np.arange(0, 1 + 1 / quantiles, 1 / quantiles)
    #     )
    #
    # @staticmethod
    # def calc_percent_rank(series: pd.Series) -> pd.Series:
    #     return series.rank(
    #         pct=True,
    #         method='dense'
    #     )
    #
    # @staticmethod
    # def calc_true_positive(predicted: pd.Series, observed: pd.Series):
    #     return np.where((observed == 1) & (predicted == 1), 1, 0)
    #
    # @staticmethod
    # def calc_false_positive(predicted: pd.Series, observed: pd.Series):
    #     return np.where((observed == 0) & (predicted == 1), 1, 0)
    #
    # @staticmethod
    # def calc_true_negative(predicted: pd.Series, observed: pd.Series):
    #     return np.where((observed == 0) & (predicted == 0), 1, 0)
    #
    # @staticmethod
    # def calc_false_negative(predicted: pd.Series, observed: pd.Series):
    #     return np.where((observed == 1) & (predicted == 0), 1, 0)

    def get_lift(self):
        """
        Lift charts show how well a model is at splitting/separating the high and low observed_validation responses.
        They are particularly useful when you are trying to define a segment using the model, and need to consider
        the trade-off between segment size and model uplift.

        Lift charts are created by:
            1. Ordering the model file (or test_2 dataset) by predicted_validation value
            2. Grouping the data into percentiles (based on predicted_validation value)
            3. Dividing the observed_validation average for the percentile by the overall observed_validation average

        i.e. lift is defined as the mean observed_validation (0 or 1) of a quantile over the population mean observed_validation

        Lift is a measure of how well predictions in a quantile perform compared to a random guess, e.g.
        predictions in the 90 percentile are 3x better than a random guess

        Cumulative lift can be interpreted as showing the increase in response rate from the model compared to taking a
        simple random sample. For example, the orange line shows that at the 80th percentile, the top 20% of
        observations (according to the model) have an average response which is 1.4 times greater than the data average.
        Whereas lift,the blue line, shows that in the 90th percentile only, the observations have a response rate
        of 1.35 times the data average

         .. plot::

            from source.pyplots.gains import pvo_classification
            pvo_classification.plot_lift()

        """
        quantile_mean = (
            self
                .observed
                .groupby(self.quantile_mid)
                .mean()
                .sort_index(ascending=False)
        )

        population_mean = (
            self
                .observed
                .mean()
        )

        lift = quantile_mean / population_mean

        return lift

    def get_cumulative_lift(self):
        """

        Cumulative lift is defined as mean lift for a given quantile and all higher quantiles

        Cumulative lift will equal lift in the highest quantile
        Cumulative lift will tend to 1 as we tend to the lowest quantile

        .. plot::

            from source.pyplots.gains import pvo_classification
            pvo_classification.plot_lift()

        """
        return (
            self
                .get_lift()
                .expanding()
                .mean()
        )

    def get_gains(self):
        """
        Gains curves are a method used to determine how good a model is at ranking data. They represent the percentage
        of the (observed_validation) response that is captured in the highest ranked portion of the data (according to
         the model).

        A useless model will provide a random prediction which means that in each 10% of the data ranked by the model
        we will get 10% of the responses. The perfect model is determined based on ranking the data by the actual
        response levels (a perfect ranking)

        Gains is defined as cumulative total number of positive observations in a given quantile and all higher
        prediction quantiles over the total number of expected positive observations under a useless model

        Gains tends to 1 as we tend to the lowest quantile

        The main use of a gains curve is to compare competing models. When comparing multiple models, the model with a
        gains curve closest to the perfect curve indicates the best model performer

        Gains is often used in marketing use cases. E.g. if we select the top 10% of customers, we will get 50% of
        our expected responses

        .. plot::

            from source.pyplots.gains import pvo_classification
            pvo_classification.plot_gains()

        """
        responses_in_quantile = (
            self
                .observed
                .groupby(self.quantile_mid)
                .sum()
                .sort_index(ascending=False)
                .expanding()
                .sum()
        )

        responses_in_population = (
            self
                .observed
                .sum()
        )

        return responses_in_quantile / responses_in_population

    def get_normalised_gini(self):
        return (
                self.calculate_gini(self.observed, self.predicted) /
                self.calculate_gini(self.observed, self.observed)
        )

    def get_gini(self):
        return self.calculate_gini(
            observed=self.observed,
            predicted_probability=self.predicted
        )

    def get_auc(self):
        return roc_auc_score(
            y_true=self.observed,
            y_score=self.predicted
        )

    def get_log_loss(self):
        return log_loss(
            y_true=self.observed,
            y_pred=self.predicted
        )

    def predict_label_using_threshold(self, threshold=0.5):
        return np.where(self.predicted > threshold, 1, 0)

    def get_precision(self, threshold=0.5):
        return precision_score(
            y_true=self.observed,
            y_pred=self.predict_label_using_threshold(threshold=threshold)
        )

    def get_recall(self, threshold=0.5):
        return recall_score(
            y_true=self.observed,
            y_pred=self.predict_label_using_threshold(threshold=threshold)
        )

    def get_metrics(self, threshold=0.5):
        return {
            'gini': self.get_gini(),
            'normalised_gini': self.get_normalised_gini(),
            'auc': self.get_auc(),
            'log_loss': self.get_log_loss()
            # 'precision': self.get_precision(),
            # 'recall': self.get_recall(),
            # 'label_prediction_threshold': threshold
        }

    @staticmethod
    def calculate_gini(observed, predicted_probability):
        assert (len(observed) == len(predicted_probability))
        all = np.asarray(np.c_[observed, predicted_probability, np.arange(len(observed))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses

        giniSum -= (len(observed) + 1) / 2.
        return giniSum / len(observed)

    def plot_gains(self):
        """

        Plots gains curve

        see :meth:`get_gains`

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.invert_xaxis()

        gains = self.get_gains().sort_index(ascending=False)

        sns.lineplot(
            x=gains.index,
            y=gains,
            ax=ax,
            label='gains curve'

        )
        sns.lineplot(
            x=np.array(gains.index.values),
            y=np.array(gains.sort_index().index.values),
            ax=ax,
            label='random guess'
        )

        ax.set_xlabel('quantile')
        ax.set_ylabel('gains')
        ax.set_title("gains")
        ax.legend()

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename=f'gains_{self.quantiles}'
        )

    def plot_lift(self):
        """

        Plots lift curve

        see :meth:`get_lift`

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.invert_xaxis()

        lift = self.get_lift()
        cumulative_lift = self.get_cumulative_lift()

        sns.lineplot(
            x=lift.index,
            y=lift,
            ax=ax,
            label='lift'

        )
        sns.lineplot(
            x=cumulative_lift.index,
            y=cumulative_lift,
            ax=ax,
            label='cumulative lift'
        )

        ax.set_xlabel('quantile')
        ax.set_ylabel('lift and cumulative lift')
        ax.set_title("lift and cumulative lift")
        ax.legend()

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename=f'lift_{self.quantiles}'
        )

    def plot_zero_one_distribution(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.distplot(
            a=self.predicted[self.observed == 1],
            bins=self.quantiles,
            norm_hist=True,
            kde=False,
            label='observed_validation 1',
            ax=ax
        )

        sns.distplot(
            a=self.predicted[self.observed == 0],
            bins=self.quantiles,
            kde=False,
            norm_hist=True,
            label='observed_validation 0',
            ax=ax
        )

        ax.set_xlabel('predicted probability')
        ax.set_ylabel('probability')
        ax.set_title("predicted probability for 0 and 1 observations")
        ax.legend()

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename=f'zero_one_distribution_{self.quantiles}'
        )

    def display_all_prediction_information_plots(self):
        self.plot_pvo()
        self.plot_lift()
        self.plot_gains()
        self.plot_zero_one_distribution()
        self.plot_pvo_quantile_error()


class PredictedAndObservedRegression(PredictedAndObserved):

    def _init__(self, predicted, observed, quantiles: int, output_dir: str = None):
        super()._init__(predicted=predicted, observed=observed, quantiles=quantiles, output_dir=output_dir)

    def get_mse(self):
        return np.mean(
            np.square(self.get_prediction_error())
        )

    def get_rmse(self):
        return np.sqrt(self.get_mse())

    def get_mae(self):
        return np.mean(
            np.abs(self.get_prediction_error())
        )

    def get_metrics(self):
        return {
            'rmse': self.get_rmse(),
            'mse': self.get_mse(),
            'mae': self.get_mae(),
        }

    def plot_predicted_and_observed(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.scatterplot(
            x=self.predicted,
            y=self.observed
        )
        sns.set_style("darkgrid")
        sns.lineplot(
            np.linspace(self.observed.min(), self.observed.max(), 100),
            np.linspace(self.observed.min(), self.observed.max(), 100),
            color='r'
        )

        ax.set_xlabel('predicted')
        ax.set_ylabel('observed')
        ax.set_title("predicted and observed")

        self.save_figure_if_output_dir_exists(
            fig=fig,
            filename=f'predicted_and_observed_{self.quantiles}'
        )


class BaselineWithUpliftPvO(ModelOutput):

    def __init__(
            self,
            baseline_pvo: PredictedAndObservedClassification,
            uplift_pvo: PredictedAndObservedClassification
    ):
        super().__init__()
        self.baseline_pvo = baseline_pvo
        self.uplift_pvo = uplift_pvo

    def get_metric_uplift(self, metric: str):
        uplift_model_metric = self.uplift_pvo.get_metrics().get(metric)
        baseline_model_metric = self.baseline_pvo.get_metrics().get(metric)

        return {
            'metric': metric,
            'baseline_model_metric': baseline_model_metric,
            'uplift_model_metric': uplift_model_metric,
            'absolute_uplift': uplift_model_metric - baseline_model_metric,
            'relative_uplift': self.uplift_pvo.get_metrics().get(metric) / self.baseline_pvo.get_metrics().get(metric)
        }

    def get_relative_uplift_by_quantile(self):
        return self.uplift_pvo.get_cumulative_lift() / self.baseline_pvo.get_cumulative_lift()

    def plot_baseline_and_uplift_one_distribution(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.distplot(
            self.baseline_pvo.predicted[self.baseline_pvo.observed == 1],
            norm_hist=True,
            ax=ax,
            label='baseline'
        )

        sns.distplot(
            self.uplift_pvo.predicted[self.uplift_pvo.observed == 1],
            norm_hist=True,
            ax=ax,
            label='uplfit'
        )

        ax.set_xlabel('probability')
        ax.set_ylabel('density')
        ax.set_title("baseline vs. uplift probability 1 distributions")
        ax.legend()

    def plot_baseline_and_uplift_zero_distribution(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.distplot(
            self.baseline_pvo.predicted[self.baseline_pvo.observed == 0],
            norm_hist=True,
            ax=ax,
            label='baseline'
        )

        sns.distplot(
            self.uplift_pvo.predicted[self.uplift_pvo.observed == 0],
            norm_hist=True,
            ax=ax,
            label='uplift'
        )

        ax.set_xlabel('probability')
        ax.set_ylabel('density')
        ax.set_title("baseline vs. uplift probability 0 distributions")
        ax.legend()

# TODO extent framework to multiclass classification - would this be a list of [predictionAndObservedMetrics?]

# TODO add chi-squared test_2 or goodness of fit for PvO
# TODO add kolmogrov smirnov test_2 for goodness of fit for PvO
# TODO plot residuals of quantiles (should expect normal around 0)
# TODO create sum of quantile residuals to determine if the are approx. normal - else systematic bias
