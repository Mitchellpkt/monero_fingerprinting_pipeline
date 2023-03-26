import pandas as pd
from typing import List, Set
from loguru import logger


def add_labels(df: pd.DataFrame, version: int = 1) -> pd.DataFrame:
    # Weak versioning for now; filters will be abstracted and version controlled later
    if version != 1:
        raise NotImplementedError("Only version 1 is currently supported")

    # Init
    initial_cols: Set[str] = set(df.keys())

    # Unlock time logic
    logger.info("Starting unlock time logic...")
    df["anomalous_unlock_time_high"] = df["unlock_time"] > 1_400_000_000
    ...

    # Fees logic
    logger.info("Starting fees logic...")
    df["anomalous_fee_zero"] = df["txn_fee_atomic"] == 0
    ...

    # etc..

    # Finalize
    logger.info("Finalizing...")
    cols_diff: List[str] = list(set(df.keys()) - initial_cols)
    df["any_anomaly"] = df[cols_diff].any(axis=1)
    return df


# Inline demo:
if __name__ == "__main__":
    data_filepath: str = "~/Projects/GitHub/monero_fingerprinting_pipeline/output/rs16_era_1679780030.csv"
    logger.info(f"Loading {data_filepath}...")
    df: pd.DataFrame = pd.read_csv(data_filepath)
    logger.info(f"Loaded {len(df)} rows. Beginning labeling")
    df = add_labels(df)
    print(df.head())
