start_height: int = 2_440_000  # 2021-09-01
end_height: int = 2_885_000  # 2023-05-13
batch_size_blocks: int = 10_000

from interface_logic import RunConfig
import server_config
from loguru import logger
import time

logger.info("Starting Sept Y21 to May Y23 run...")
tic: float = time.perf_counter()

current_height = start_height
while current_height < end_height:
    next_height = min(current_height + batch_size_blocks, end_height)
    output_file_path: str = f"/home/user/isthmus_dev/monero_fingerprinting_pipeline/output/rs_S21_to_My23_{current_height}_{next_height}.feather"
    r = RunConfig(
        url=server_config.url,
        port=server_config.port,
        verbose=True,
        num_workers=16,
        sleep_for_rate_limiting_sec=0.5,
        transaction_batch_size=250,
        start_height=current_height,
        end_height=next_height,
        save_to_feather=output_file_path,
        run_on_init=True,
    )
    current_height = next_height
    logger.info(f"\nDone with height range {current_height} to {next_height}")
    logger.info(
        f"Took {time.perf_counter() - tic:.2f} seconds to run over {next_height-current_height} blocks."
    )
    logger.info(f"Output saved to {output_file_path}")

logger.info(f"\nDone!")
logger.info(f"Took {time.perf_counter() - tic:.2f} seconds to run over {end_height-start_height} blocks.")
logger.info(f"Output saved to {output_file_path}")
