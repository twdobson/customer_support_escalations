import os
import pandas as pd


class ModelOutput:

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.output_dir_exists = output_dir is not None

        try:
            os.makedirs(output_dir)
        except:
            print("output dir either exists or was not provided")

    def save_figure_if_output_dir_exists(self, fig, filename: str):
        if self.output_dir_exists:
            fig.savefig(
                os.path.join(self.output_dir, 'figures', f"{filename}.png")
            )

    def write_df_if_output_dir_exists(self, df: pd.DataFrame, filename: str):
        if self.output_dir_exists:
            df.to_excel(
                os.path.join(self.output_dir, 'data', filename + '.xlsx'),
                sheet_name=filename
            )
