# ## Import libraries

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import pandas as pd
from isthmuslib import VectorMultiset, process_queue
from tqdm.auto import tqdm
from loguru import logger


def flatten_ring_members(row: pd.Series) -> pd.Series:
    """
    Function to flatten ring members.

    :param row: A row from a DataFrame
    :type row: pd.Series
    :return: Row with flattened ring members
    :rtype: pd.Series
    """
    if isinstance(row["inputs"], str):
        input_str: str = row["inputs"].replace("'", '"')
        inputs_data: List[Dict[str, Any]] = json.loads(input_str)
    else:
        inputs_data = row["inputs"]
    flat_rings: List[int] = []
    for i_ in inputs_data:
        flat_rings.extend(i_["ring_member_indices"])
    row["flat_ring_members"] = flat_rings
    return row


def process_file(file: Path, flattened_dir: Path, verbose: bool = True) -> None:
    """
    Function to process a single file.

    :param file: File path
    :type file: Path
    :param flattened_dir: Directory path for flattened files
    :type flattened_dir: Path
    :param verbose: Flag to print messages, defaults to True
    :type verbose: bool, optional
    """
    if verbose:
        logger.info(f"Processing file {file}")
    txns: VectorMultiset = VectorMultiset().read_any(file, inplace=False)
    txns.data["flat_ring_members"] = None
    txns.data = txns.data.apply(flatten_ring_members, axis=1, raw=False)
    flattened_file = flattened_dir / f"{file.stem}_flattened.feather"
    columns_to_keep = [
        "tx_hash",
        "block_height",
        "block_timestamp",
        "unlock_time",
        "num_inputs",
        "num_outputs",
        "txn_fee_atomic",
        "flat_ring_members",
    ]
    txns.data = txns.data[columns_to_keep]
    txns.data.to_feather(flattened_file)
    if verbose:
        print(f"Processed and saved file {flattened_file}")


def flatten_files(
    target_dir: Union[Path, str], verbose: bool = True, num_workers: Optional[int] = None
) -> None:
    """
    Function to flatten files.

    :param target_dir: Directory path
    :type target_dir: Union[Path, str]
    :param verbose: Flag to print messages, defaults to True
    :type verbose: bool, optional
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {target_dir} not found.")

    flattened_dir = target_dir / "flattened"
    flattened_dir.mkdir(parents=True, exist_ok=True)

    feather_files = list(target_dir.glob("*.feather"))
    iterable: List[Dict[str, Any]] = [
        {"file": file, "flattened_dir": flattened_dir, "verbose": verbose} for file in feather_files
    ]

    process_queue(
        func=process_file,
        iterable=iterable,
        pool_function="kwargsmap",
        num_workers=num_workers,
    )


def concatenate_flattened_files(target_dir: Union[Path, str], verbose: bool = True) -> pd.DataFrame:
    """
    Function to concatenate flattened files.

    :param target_dir: Directory path
    :type target_dir: Union[Path, str]
    :param verbose: Flag to print messages, defaults to True
    :type verbose: bool, optional
    :return: Concatenated DataFrame
    :rtype: pd.DataFrame
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {target_dir} not found.")

    feather_files = list(target_dir.glob("*.feather"))

    dfs: List[pd.DataFrame] = []
    for file in tqdm(feather_files, desc="Concatenating files", mininterval=1):
        df = pd.read_feather(file)
        dfs.append(df)
        if verbose:
            print(f"Loaded file {file}")

    result_df = pd.concat(dfs, ignore_index=True)

    return result_df


if __name__ == "__main__":
    num_workers: int = 1
    original_dir: str = "/home/m/Projects/GitHub/monero_fingerprinting_pipeline/output"
    logger.info(f"Flattening files in {original_dir}...")
    flatten_files(original_dir, num_workers=num_workers)
    logger.info(f"Concatenating flattened files in {original_dir}/flattened...")
    df: pd.Dataframe = concatenate_flattened_files(Path(original_dir) / "flattened")
    final_output_path: Path = Path(original_dir) / "flattened" / "all" / "transactions_flattened.feather"
    logger.info(f"Saving concatenated DataFrame to {final_output_path}...")
    df.to_feather(final_output_path)
