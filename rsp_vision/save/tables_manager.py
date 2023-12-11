from pathlib import Path

import pandas as pd


class AnalysisSuccessTable:
    def __init__(self, path: Path):
        self.path = path / "analysis_success.log"

        if not self.path.exists():
            self.df = pd.DataFrame(
                columns=["dataset_name", "date", "latest_job_id", "error"]
            )
            self.df.to_csv(self.path)
        else:
            self.df = pd.read_csv(self.path, index_col=0, header=0)

    def update(
        self, dataset_name: str, date: str, latest_job_id: int, error: str
    ):
        if not self.is_dataset_in_table(dataset_name):
            self.add(dataset_name, date, latest_job_id, error)
        else:
            self.df.loc[self.df["dataset_name"] == dataset_name] = {
                "dataset_name": dataset_name,
                "date": date,
                "latest_job_id": latest_job_id,
                "error": error,
            }
            self.df.to_csv(self.path)

    def add(
        self, dataset_name: str, date: str, latest_job_id: int, error: str
    ):
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {
                        "dataset_name": dataset_name,
                        "date": date,
                        "latest_job_id": latest_job_id,
                        "error": error,
                    },
                    index=[0],
                ),
            ]
        )
        self.df.to_csv(self.path)

    def is_dataset_in_table(self, dataset_name: str) -> bool:
        return dataset_name in self.df["dataset_name"]
