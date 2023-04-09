import pandas as pd
from typing import List, Set
from loguru import logger
from tqdm.auto import tqdm
from isthmuslib import process_queue, get_num_workers

delay_tolerance_blocks: float = 720
unlock_time_high_cutoff: int = 100_000_000


def get_len_extra_items(x: str) -> int:
    return x.count(",") + 1


def ringxor_anomaly(t) -> bool:
    inputs = eval(t)  # yolo, hope your node isn't evil

    ringsize_minus_1: int = len(inputs[0]["ring_member_indices"]) - 1
    if len(inputs) == 1:
        return False

    rings: List[Set[int]] = [set(r_["ring_member_indices"]) for r_ in inputs]
    for i_left in range(len(rings)):
        for i_right in range(i_left + 1, len(rings)):
            if i_right >= i_left:
                if len(rings[i_left].intersection(rings[i_right])) == ringsize_minus_1:
                    return True

    return False


def batch_ringxor_anomaly_core_worker(inputs_data_raw: List[str]) -> List[bool]:
    return [ringxor_anomaly(t) for t in tqdm(inputs_data_raw)]


def batch_ringxor_anomaly(inputs_data_raw: List[str], num_workers: int = None) -> List[bool]:
    if num_workers is None:
        num_workers: int = get_num_workers(num_workers)

    # Estimate batch size
    batch_size: int = len(inputs_data_raw) // num_workers + 1
    batches: List[List[str]] = [
        inputs_data_raw[i : i + batch_size] for i in range(0, len(inputs_data_raw), batch_size)
    ]

    # Sanity check
    if not sum([len(b) for b in batches]) == len(inputs_data_raw):
        raise ValueError(
            "Batching bug, sum of batch sizes does not equal total number of inputs, please debug"
        )

    # Process the queue
    results: List[List[bool]] = process_queue(
        func=batch_ringxor_anomaly_core_worker,
        iterable=batches,
        num_workers=num_workers,
        pool_function="map",
    )

    return [item for sublist in results for item in sublist]


def add_labels(df: pd.DataFrame, version: int = 1, num_workers: int = None) -> pd.DataFrame:
    # Weak versioning for now; filters will be abstracted and version controlled later
    if version != 1:
        raise NotImplementedError("Only version 1 is currently supported")

    # Init
    logger.info("Initializing...")
    initial_cols: Set[str] = set(df.keys())

    # Heuristic: Unlock times above 100,000,000 are atypical
    logger.info("Analyzing unlock times (high) ...")
    df["anomalous_unlock_time_high"] = df["unlock_time"] > unlock_time_high_cutoff

    # Heuristic: Unlock times more than 720 blocks under the current block height are atypical
    # This could mean they were created with a low value or delayed
    # Note that unlock_time==0 is OK, this is the standard wallet2 behavior if not specified
    logger.info("Analyzing unlock times (low) ...")
    df["anomalous_unlock_time_low"] = df["unlock_time"] + delay_tolerance_blocks < df["block_height"]
    df["anomalous_unlock_time_low"] = df["anomalous_unlock_time_low"] & (df["unlock_time"] != 0)

    # Heuristic: fees should not be 0
    logger.info("Analyzing fees ...")
    df["anomalous_fee_zero"] = df["txn_fee_atomic"] == 0

    # Heuristic: len(tx_extra) > 1500 [in units of list length] is atypical
    logger.info("Analyzing extra length ...")
    df["len_extra"] = df["extra"].apply(get_len_extra_items)
    df["anomalous_extra_very_long"] = df["len_extra"] > 1500

    # Analyze rings
    logger.info("Analyzing intra transaction inter ring decoy reuse ...")
    df["anomalous_ringxor"] = batch_ringxor_anomaly(df["inputs"].tolist(), num_workers=num_workers)

    # Finalize
    logger.info("Finalizing...")
    cols_diff: List[str] = list(set(df.keys()) - initial_cols - {"len_extra"})
    df["any_anomaly"] = df[cols_diff].any(axis=1)

    # Print some stats
    seperator: str = "\n" + 15 * "_"
    s: str = f"{seperator}\nDescription:\n\n"
    s += f"Number of transactions: {len(df)}\n"
    s += f"Block height range: {df['block_height'].min()} - {df['block_height'].max()}"
    s += f"\n{seperator}\nStats (per anomaly):\n"
    for col in cols_diff:
        s += f"\n{col}: {df[col].sum()} anomalies"
    s += f"\n{seperator}\nTotal anomalous transactions: {df['any_anomaly'].sum()}"
    logger.info(s)

    return df


# Inline demo:
if __name__ == "__main__":
    data_filepath: str = "~/Projects/GitHub/monero_fingerprinting_pipeline/output/rs16_era_1680989344.csv"
    logger.info(f"Loading {data_filepath}...")
    df: pd.DataFrame = pd.read_csv(data_filepath)
    logger.info(f"Loaded {len(df)} rows. Beginning labeling")
    df = add_labels(df, num_workers=8)

    # print some ringxor examples
    print(df[df["anomalous_ringxor"]].head()["tx_hash"].tolist())

    # print(df.head())
