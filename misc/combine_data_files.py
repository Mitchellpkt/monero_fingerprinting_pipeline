import os
import pandas as pd
from typing import List, Any
import isthmuslib as isli


def csv_wrapper(filename: Any) -> pd.DataFrame:
    print(f"... loading {filename}")
    return pd.read_csv(filename)


def combine_data_files(input_directory: str, output_filename: str) -> None:
    """
    Combine all the feather files in a given directory into a single feather file.
    :param input_directory: The directory to look for data files.
    :param output_filename: The name of the output feather file.
    """
    # List comprehension to get all files in the directory
    files: List[str] = [
        os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".feather")
    ]  # sic

    # Read each feather file and concatenate it into the combined DataFrame
    dfs: List[pd.DataFrame] = isli.process_queue(
        func=csv_wrapper,
        iterable=files,
        pool_function="map",
    )
    combined_df: pd.DataFrame = pd.concat(dfs)
    isli.convert_dtypes_advanced(combined_df, inplace=True)

    # Write the combined DataFrame to a new feather file
    combined_df.reset_index(drop=True).to_feather(output_filename)
    print(f"Combined DataFrame written to {output_filename}")


if __name__ == "__main__":
    # Combine the feather files in the data directory into a single feather file
    combine_data_files(
        "/home/bird/data_drive/monero/output_raw_etl",
        "/home/bird/Projects/GitHub/monero_fingerprinting_pipeline/DATA/transactions.feather",
    )
