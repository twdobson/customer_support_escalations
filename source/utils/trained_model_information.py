import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from source.utils.model_output import ModelOutput


class TrainedModelInformation(ModelOutput):

    def __init__(self, model, feature_columns: list, output_dir: str = None):
        super().__init__(output_dir=output_dir)
        self.model = model
        self.feature_columns = feature_columns

    def get_feature_importance(self):
        if type(self.model) == lgb.basic.Booster:
            return self._get_lgb_feature_importance()
        else:
            pass

    def _get_lgb_feature_importance(self) -> pd.DataFrame:
        gain_importance = pd.Series(self.model.feature_importance('gain'))
        split_importance = pd.Series(self.model.feature_importance('split'))
        return pd.DataFrame({
            'feature': self.feature_columns,
            'gain_feature_importance': gain_importance / gain_importance.max(),
            'split_feature_importance': split_importance / split_importance.max(),
            'gain_rank': gain_importance.rank(ascending=False, method="dense"),
            'split_rank': split_importance.rank(ascending=False, method="dense")
        }).sort_values('gain_feature_importance', ascending=False)

    def plot_feature_importances(self, top_n: int = 15):
        feature_importance = self.get_feature_importance()
        n_features_to_plot = min(top_n, len(self.model.feature_name()))

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.barplot(
            y=feature_importance['feature'][0:n_features_to_plot],
            x=feature_importance.iloc[:, 1][0:n_features_to_plot],
            ax=ax,
            color='b'
        )

        ax.set_xlabel(feature_importance.iloc[: 1].name)
        ax.set_ylabel('feature')
        ax.set_title("feature importance")
        ax.legend()

    def plot_test_train_evaluation_metric_by_nround(self, evals_result):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        metrics_by_train_test = pd.DataFrame(evals_result).T
        for idx, metric in enumerate(metrics_by_train_test.columns):
            for observation_set in metrics_by_train_test.index:
                evaluation_metric = pd.Series(evals_result[observation_set][metric])
                sns.lineplot(
                    x=range(evaluation_metric.shape[0]),
                    y=evaluation_metric,
                    ax=axes[idx],
                    label=observation_set + f' {metric} by n rounds')

    def plot_lgb_n_graphviz_trees(self, n: int):
        for idx in range(n):
            graph = lgb.create_tree_digraph(
                booster=self.model,
                tree_index=idx,
                show_info=['split_gain', 'leaf_count', 'internal_value'])

            graph.render(
                view=False,
                directory=self.folder_structure.dir_tree_graphviz,
                filename=str(idx) + "_tree",
                cleanup=True)

    # def write_feature_importance(self):
    #     write_csv(
    #         df=self.get_feature_importance(),
    #         dir_output=self.folder_structure.dir_data_csv,
    #         df_name='feature_importance')
