import pandas as pd
from typing import List, Set
from loguru import logger

delay_tolerance_blocks: float = 720
unlock_time_high_cutoff: int = 100_000_000


def get_len_extra_items(x: str) -> int:
    return x.count(",") + 1


def add_labels(df: pd.DataFrame, version: int = 1) -> pd.DataFrame:
    # Weak versioning for now; filters will be abstracted and version controlled later
    if version != 1:
        raise NotImplementedError("Only version 1 is currently supported")

    # Init
    initial_cols: Set[str] = set(df.keys())

    # Heuristic: Unlock times above 100,000,000 are atypical
    df["anomalous_unlock_time_high"] = df["unlock_time"] > unlock_time_high_cutoff

    # Heuristic: Unlock times more than 720 blocks under the current block height are atypical
    # This could mean they were created with a low value or delayed
    # Note that unlock_time==0 is OK, this is the standard wallet2 behavior if not specified
    df["anomalous_unlock_time_low"] = df["unlock_time"] + delay_tolerance_blocks < df["block_height"]
    df["anomalous_unlock_time_low"] = df["anomalous_unlock_time_low"] & (df["unlock_time"] != 0)

    # Heuristic: fees should not be 0
    df["anomalous_fee_zero"] = df["txn_fee_atomic"] == 0

    # Heuristic: len(tx_extra) > 1500 [in units of list length] is atypical
    df["len_extra"] = df["extra"].apply(get_len_extra_items)
    df["anomalous_extra_very_long"] = df["len_extra"] > 1500

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