import json
import pathlib
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel
import pandas as pd
import time
from tqdm.auto import tqdm


@dataclass
class JsonRpcClient(BaseModel):
    url: str
    port: int
    verbose: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.verbose:
            logger.info(f"Initializing JSON-RPC client with URL {self.url} and port {self.port}")

    def _send_request(self, method: str, params: dict) -> dict:
        if self.verbose:
            logger.info(f"Sending request: {method} with params {params}")
        headers = {"Content-Type": "application/json"}
        data = {"jsonrpc": "2.0", "id": "0", "method": method, "params": params}
        response = requests.post(f"{self.url}:{self.port}/json_rpc", headers=headers, json=data)

        if response.status_code == 200:
            if self.verbose:
                logger.info(f"Request successful: {method}")
            return response.json()["result"]
        raise Exception(f"Error: {response.status_code} - {response.text}")

    def call(self, method: str, params: Optional[Union[Dict, List]] = None) -> Any:
        if params is None:
            params = {}
        payload = {
            "jsonrpc": "2.0",
            "id": "0",
            "method": method,
            "params": params,
        }
        if self.verbose:
            logger.info(f"Sending request: {method} with params {params}")
        response = requests.post(f"{self.url}:{self.port}/json_rpc", json=payload)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise Exception(f"JSON-RPC error: {result['error']}")
        return result["result"]


class ConnectionConfig(BaseModel):
    """This is the configuration for a connection to a Monero node"""

    # Must include URL and port at initialization
    url: str
    port: int
    verbose: bool = True
    num_workers: int = 16
    sleep_for_rate_limiting_sec: Optional[float] = 0.1
    transaction_batch_size: int = 250

    # Internally-managed
    full_url: str = None
    json_rpc_client: JsonRpcClient = None

    def setup(self):
        self.full_url = f"{self.url}:{self.port}"
        self.json_rpc_client = JsonRpcClient(
            url=self.url,
            port=self.port,
            verbose=self.verbose,
        )
        self.num_workers = min((self.num_workers, cpu_count()))

    def __init__(self, **data):
        super().__init__(**data)
        self.setup()

    def to_connection_config_json_file(self, json_file_path: Union[pathlib.Path, str], verbose: bool = True):
        with open(json_file_path, "w") as f:
            json.dump(self.dict(), f, indent=4)
        if verbose:
            logger.info(f"ConnectionConfig saved to {json_file_path}")


def load_connection_config_from_json_file(
    json_file: Union[pathlib.Path, str], verbose: bool = True
) -> ConnectionConfig:
    if not pathlib.Path(json_file).exists():
        raise FileNotFoundError(f"File {json_file} not found")
    with open(json_file, "r") as f:
        connection_config_dict = json.load(f)
    if verbose:
        logger.info(f"ConnectionConfig loaded from {json_file}")
    return ConnectionConfig(**connection_config_dict)


def return_json_rpc_client(rpc_client: Union[JsonRpcClient, ConnectionConfig]) -> JsonRpcClient:
    """Helper function to make other inputs more flexible, ingests either a JsonRpcClient or a ConnectionConfig"""
    if isinstance(rpc_client, ConnectionConfig):
        return rpc_client.json_rpc_client
    elif isinstance(rpc_client, JsonRpcClient):
        return rpc_client
    else:
        raise TypeError(
            f"Unknown input type {type(rpc_client)=}; please provide JsonRpcClient or ConnectionConfig"
        )


def get_block_count(rpc_client: Union[JsonRpcClient, ConnectionConfig]) -> int:
    json_rpc_client: JsonRpcClient = return_json_rpc_client(rpc_client)
    return json_rpc_client.call("get_block_count")["count"]


def get_block(rpc_client: Union[JsonRpcClient, ConnectionConfig], height: int) -> Dict[str, Any]:
    json_rpc_client: JsonRpcClient = return_json_rpc_client(rpc_client)
    return json_rpc_client.call("get_block", {"height": height})


def get_transactions_raw(
    txs_hashes: List[str],
    url: str,
    port: int,
    decode_as_json: bool = True,
    prune: bool = True,
    split: bool = True,
    **_,
) -> Dict[str, Any]:
    """
    Retrieve rawish transactions data from a Monero node.

    :param txs_hashes: list of transaction hashes to retrieve
    :param url: daemon url
    :param port: daemon port
    :param decode_as_json: (passed directly in API call)
    :param prune: (passed directly in API call)
    :param split: (passed directly in API call)
    :return: rawish transaction data sent back by the node
    """
    method = "get_transactions"
    full_url = f"{url}:{port}/{method}"

    params = {
        "txs_hashes": txs_hashes,
        "decode_as_json": decode_as_json,
        "prune": prune,
        "split": split,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(full_url, data=json.dumps(params), headers=headers)

    if response.status_code != 200:
        raise Exception(f"Request error: {response.status_code}, {response.text}")

    return response.json()


def extract_transactions_data(json_data: Dict[str, any]) -> List[Dict[str, Any]]:
    """
    Extracts and flattens some fields of interest from the transaction data sent back by node.

    :param json_data: Node response, for example the response of get_transactions
    :return: a list of dictionaries, one per transaction, with the following fields:
        - tx_hash
        - block_height
        - block_timestamp
        - version
        - unlock_time
        - num_inputs
        - num_outputs
        - txn_fee_atomic
        - rct_type
        - extra
    """
    transactions = json_data["txs"]
    extracted_data = []

    for tx in transactions:
        tx_hash = tx["tx_hash"]
        block_height = tx["block_height"]
        block_timestamp = tx["block_timestamp"]
        json_data = json.loads(tx["as_json"])

        extracted_data.append(
            {
                "tx_hash": tx_hash,
                "block_height": block_height,
                "block_timestamp": block_timestamp,
                "version": json_data["version"],
                "unlock_time": json_data["unlock_time"],
                "num_inputs": len(json_data["vin"]),
                "num_outputs": len(json_data["vout"]),
                "txn_fee_atomic": json_data["rct_signatures"]["txnFee"],
                "rct_type": json_data["rct_signatures"]["type"],
                "extra": json_data["extra"],
            }
        )

    return extracted_data


def get_transactions_over_height_range_single_core(
    connection_config: ConnectionConfig,
    start_height: int,
    end_height: int,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve transaction data from a Monero node over a range of block heights.

    :param connection_config: The ConnectionConfig class above
    :param start_height: starting block height
    :param end_height: ending block height
    :param verbose: whether to display a progress bar
    :return: transaction data sent back by the node (list of dicts, 1 element per transaction)
    """

    txs_hashes = []
    for height in range(start_height, end_height + 1):
        if verbose:
            logger.info(f"Processing block {height}", end="\r")
        block = get_block(connection_config, height)
        if connection_config.sleep_for_rate_limiting_sec:
            time.sleep(connection_config.sleep_for_rate_limiting_sec)
        if "tx_hashes" in block and len(block["tx_hashes"]):
            txs_hashes.extend(block["tx_hashes"])

    if verbose:
        logger.info(
            f"About to retrieve {len(txs_hashes)} transactions from blocks {start_height} to {end_height}"
        )

    # If no batching, or batch > hashes, return in a single shot
    if not connection_config.transaction_batch_size or connection_config.transaction_batch_size >= len(
        txs_hashes
    ):
        return extract_transactions_data(get_transactions_raw(txs_hashes, **connection_config.dict()))

    # Use np.array_split to chunk up the hashes into batches
    txs_hash_batches = list(
        list(x) for x in np.array_split(txs_hashes, connection_config.transaction_batch_size)
    )
    result: List[Dict[str, Any]] = []
    p = (
        tqdm(txs_hash_batches, desc="Core transaction pagination", total=len(txs_hash_batches))
        if verbose
        else txs_hash_batches
    )
    for txs_hash_batch in p:
        result.extend(
            extract_transactions_data(get_transactions_raw(txs_hash_batch, **connection_config.dict()))
        )
        if connection_config.sleep_for_rate_limiting_sec:
            time.sleep(connection_config.sleep_for_rate_limiting_sec)
    return result


# def get_transactions_over_height_range_single_core(
#     connection_config: ConnectionConfig,
#     start_height: int,
#     end_height: int,
#     verbose: bool = True,
# ) -> List[Dict[str, Any]]:
#     """
#     Retrieve transaction data from a Monero node over a range of block heights.
#
#     :param connection_config: The ConnectionConfig class above
#     :param start_height: starting block height
#     :param end_height: ending block height
#     :param verbose: whether to display a progress bar
#     :return: transaction data sent back by the node (list of dicts, 1 element per transaction)
#     """
#
#     result: List[Dict[str, Any]] = []
#     for height in range(start_height, end_height + 1):
#         if verbose:
#             logger.info(f"Processing block {height}", end="\r")
#         block = get_block(connection_config, height)
#         if connection_config.sleep_for_rate_limiting_sec:
#             time.sleep(connection_config.sleep_for_rate_limiting_sec)
#
#         if "tx_hashes" not in block:
#             continue
#         if not len(block["tx_hashes"]):
#             continue
#
#         txs_hashes = block["tx_hashes"]
#
#         if verbose:
#             logger.info(
#                 f"About to retrieve {len(txs_hashes)} transactions from blocks {start_height} to {end_height}"
#             )
#
#         # If no transaction batching, or batch size > request size, return in a single shot
#         if not connection_config.transaction_batch_size or connection_config.transaction_batch_size >= len(
#             txs_hashes
#         ):
#             result.extend(
#                 extract_transactions_data(get_transactions_raw(txs_hashes, **connection_config.dict()))
#             )
#
#         # In the event of multiple batches, use np.array_split to chunk up the hashes into batches
#         txs_hash_batches = (
#             list(x) for x in np.array_split(txs_hashes, connection_config.transaction_batch_size)
#         )
#         for txs_hash_batch in txs_hash_batches:
#             raw_txs = get_transactions_raw(txs_hash_batch, **connection_config.dict())
#             # Debug
#             try:
#                 extracted_data = extract_transactions_data(raw_txs)
#             except Exception as e:
#                 logger.error(f"Height {height}: Error extracting data from\n{txs_hash_batches=}\n{raw_txs=}")
#                 raise e
#             result.extend(extracted_data)
#             if connection_config.sleep_for_rate_limiting_sec:
#                 time.sleep(connection_config.sleep_for_rate_limiting_sec)
#
#     return result


def get_transactions_over_height_range(
    connection_config: ConnectionConfig,
    start_height: int,
    end_height: int,
    verbose: bool = True,
    save_to_csv: Optional[Union[str, pathlib.Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve rawish transactions data from a Monero node over a range of block heights.

    :param connection_config: The ConnectionConfig class above
    :param start_height: starting block height
    :param end_height: ending block height
    :param verbose: whether to display a progress bar
    :param save_to_csv: if provided, save the data to a csv file
    :return: rawish transaction data sent back by the node
    """
    if verbose:
        logger.info(f"Processing blocks {start_height} to {end_height} with {connection_config.num_workers}")

    # Break up the height range into num_workers number of chunks as evenly as possible
    height_range_iterables = [
        (x[0], x[-1])
        for x in np.array_split(range(start_height, end_height + 1), connection_config.num_workers)
        if len(x)
    ]

    with Pool(min((connection_config.num_workers, len(height_range_iterables)))) as p:
        txs_data_nested: List[List[Dict[str, Any]]] = p.starmap(
            get_transactions_over_height_range_single_core,
            [
                (connection_config, start_height, end_height, verbose)
                for start_height, end_height in height_range_iterables
            ],
        )
    transactions: List[Dict[str, Any]] = [tx for txs in txs_data_nested for tx in txs]
    if save_to_csv is not None:
        transactions_to_dataframe(
            transactions, save_to_csv=save_to_csv, verbose=verbose, return_df=False
        )  # This saves it to a CSV file
    return transactions


def transactions_to_dataframe(
    txs_data: List[Dict[str, Any]],
    save_to_csv: Optional[Union[str, pathlib.Path]] = None,
    verbose: bool = True,
    return_df: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Convert transaction data to a Pandas DataFrame.

    :param txs_data: transaction data (list of dicts, 1 element per transaction)
    :param save_to_csv: optional path to save the data as a CSV file
    :param verbose: whether to display a progress bar
    :return: Pandas DataFrame
    """
    df: pd.DataFrame = pd.DataFrame(txs_data)
    if save_to_csv is not None:
        pathlib.Path(save_to_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to_csv, index=False)
        if verbose:
            logger.info(f"Saved to {save_to_csv}")
    if return_df:
        return df


class RunConfig(ConnectionConfig):
    """Helper object for managing long passes over the chain"""

    start_height: int = 0
    end_height: int = None
    save_to_csv: Optional[Union[str, pathlib.Path]] = None
    run_on_init: bool = True

    # Inherits:
    # url: str
    # port: int
    # verbose: bool = True
    # num_workers: int = 16
    # full_url: str = None
    # json_rpc_client: JsonRpcClient = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.run_on_init:
            self.run()

    def run(self):
        if self.end_height is None:
            self.end_height = get_block_count(self)
        get_transactions_over_height_range(
            connection_config=self,
            start_height=self.start_height,
            end_height=self.end_height,
            verbose=True,
            save_to_csv=self.save_to_csv,
        )

    def to_run_config_json_file(self, file_path: Union[str, pathlib.Path], verbose: bool = True):
        with open(file_path, "w") as f:
            json.dump(self.dict(), f, indent=4)
        if verbose:
            logger.info(f"Saved to {file_path}")


def load_run_config_from_json_file(file_path: Union[str, pathlib.Path], verbose: bool = True) -> RunConfig:
    with open(file_path, "r") as f:
        run_config = RunConfig(**json.load(f))
    if verbose:
        logger.info(f"Loaded from {file_path}")
    return run_config
