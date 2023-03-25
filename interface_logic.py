import json
import pathlib
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel


@dataclass
class JsonRpcClient:
    verbose: bool = True

    def __init__(self, url: str, port: int, verbose: bool, **_):
        if self.verbose:
            logger.info(f"Initializing JSON-RPC client with URL {url} and port {port}")
        self.url = url
        self.port = port
        self.verbose = verbose

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
    """ This is the configuration for a connection to a Monero node """
    # Must include URL and port at initialization
    url: str
    port: int
    verbose: bool = True
    num_workers: int = 16

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

    def to_connection_config_json_file(self, json_file: Union[pathlib.Path, str], verbose: bool = True):
        with open(json_file, "w") as f:
            json.dump(self.dict(), f)
        if verbose:
            logger.info(f"ConnectionConfig saved to {json_file}")


def load_connection_config_from_json_file(json_file: Union[pathlib.Path, str],
                                          verbose: bool = True) -> ConnectionConfig:
    if not pathlib.Path(json_file).exists():
        raise FileNotFoundError(f"File {json_file} not found")
    with open(json_file, "r") as f:
        connection_config_dict = json.load(f)
    logger.info(f"ConnectionConfig loaded from {json_file}")
    return ConnectionConfig(**connection_config_dict)


def return_json_rpc_client(
        rpc_client: Union[JsonRpcClient, ConnectionConfig]
) -> JsonRpcClient:
    """ Helper function to make other inputs more flexible, ingests either a JsonRpcClient or a ConnectionConfig"""
    if isinstance(rpc_client, ConnectionConfig):
        return rpc_client.json_rpc_client
    elif isinstance(rpc_client, JsonRpcClient):
        return rpc_client
    else:
        raise TypeError(f"Unknown input type {type(rpc_client)=}; please provide JsonRpcClient or ConnectionConfig")


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
        *_,
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
        connection_config: ConnectionConfig, start_height: int, end_height: int, verbose: bool = True,
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
            print(f"Processing block {height}", end="\r")
        block = get_block(connection_config, height)
        txs_hashes.extend(block["tx_hashes"])

    return extract_transactions_data(get_transactions_raw(txs_hashes, **connection_config.dict()))


def get_transactions_over_height_range(
        connection_config: ConnectionConfig, start_height: int, end_height: int, verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve rawish transactions data from a Monero node over a range of block heights.

    :param connection_config: The ConnectionConfig class above
    :param start_height: starting block height
    :param end_height: ending block height
    :param verbose: whether to display a progress bar
    :return: rawish transaction data sent back by the node
    """
    if verbose:
        logger.info(f"Processing blocks {start_height} to {end_height} with {connection_config.num_workers}")

    # Break up the height range into num_workers number of chunks as evenly as possible
    height_range_iterables = [(x[0], x[-1]) for x in
                              np.array_split(range(start_height, end_height + 1), connection_config.num_workers)]

    with Pool(connection_config.num_workers) as p:
        txs_data_nested: List[List[Dict[str, Any]]] = p.starmap(
            get_transactions_over_height_range_single_core,
            [(connection_config, start_height, end_height, verbose) for start_height, end_height in
             height_range_iterables],
        )
    return [tx for txs in txs_data_nested for tx in txs]
