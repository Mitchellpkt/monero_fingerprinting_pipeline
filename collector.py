import json
import requests
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import argparse
from loguru import logger


@dataclass
class JsonRpcClient:
    verbose: bool = True

    def __init__(self, url: str, port: int, verbose: bool):
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
        else:
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


def get_block_count(json_rpc_client: JsonRpcClient) -> int:
    return json_rpc_client.call("get_block_count")["count"]


def get_block(json_rpc_client: JsonRpcClient, height: int) -> Dict[str, Any]:
    return json_rpc_client.call("get_block", {"height": height})


def get_transactions(
    txs_hashes: List[str],
    url: str,
    port: int,
    decode_as_json: bool = False,
    prune: bool = False,
    split: bool = False,
) -> List[Dict[str, Any]]:
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


def extract_transactions_data(json_data) -> List[Dict[str, Any]]:
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


def main():
    # Handle the command line arguments
    parser = argparse.ArgumentParser(description="Retrieve Monero block transactions.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1",
        help="Monero node URL (default: http://127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=18081, help="Monero node port (default: 18081)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (default: False)",
    )
    args = parser.parse_args()

    # Instantiate JSON-RPC client
    if args.verbose:
        logger.info(f"... About to instantiate JSON-RPC client with URL {args.url}")
    json_rpc_client = JsonRpcClient(args.url, args.port, args.verbose)

    # Get the latest block height
    if args.verbose:
        logger.info(f"... About to get the latest block height")
    block_count = get_block_count(json_rpc_client)

    # Retrieve the transaction hashes:
    if args.verbose:
        logger.info(f"... About to retrieve the transaction hashes")
    transactions_hashes = get_block(json_rpc_client, block_count - 1)["tx_hashes"]

    # Retrieve transactions for the latest block
    if args.verbose:
        logger.info(f"... About to retrieve transactions for the latest block (height: {block_count - 1})")
    raw_transactions = get_transactions(
        txs_hashes=transactions_hashes,
        url=args.url,
        port=args.port,
        decode_as_json=True,
        prune=True,
        split=True,
    )
    logger.info(f"Retrieved {len(raw_transactions)} transactions for block height {block_count - 1}.")

    # Extract transactions data
    transactions = extract_transactions_data(raw_transactions)
    logger.info(f"Extracted {len(transactions)} transactions data.")
    logger.info(f"\n{json.dumps(transactions, indent=4)}")


if __name__ == "__main__":
    logger.info(f"Hello! Syntax hint: python collector.py --url http://your-node-url --port 18081 --verbose")
    main()
