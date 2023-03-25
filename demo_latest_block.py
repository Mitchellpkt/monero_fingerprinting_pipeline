import argparse
import json

from loguru import logger

from interface_logic import (
    JsonRpcClient,
    get_block_count,
    get_block,
    get_transactions,
    extract_transactions_data,
)


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
        logger.info("... About to get the latest block height")
    block_count = get_block_count(json_rpc_client)

    # Retrieve the transaction hashes:
    if args.verbose:
        logger.info("... About to retrieve the transaction hashes")
    transactions_hashes = get_block(json_rpc_client, block_count - 1)["tx_hashes"]

    # Retrieve transactions for the latest block
    if args.verbose:
        logger.info("... About to retrieve transactions for the latest block (height: {block_count - 1})")
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
