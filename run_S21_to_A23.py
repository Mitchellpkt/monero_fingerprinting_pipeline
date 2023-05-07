start_height: int = 2440000  # 2021-09-01
end_height: int = 2860073  # 2023-04-08

from interface_logic import RunConfig
import server_config
from loguru import logger
import time

logger.info("Starting Sept Y21 to Apr Y23 run...")
output_file_path: str = f"/home/user/isthmus_dev/monero_fingerprinting_pipeline/output/rs_S21_to_A23{int(time.time())}.csv"
tic: float = time.perf_counter()
r = RunConfig(
    url=server_config.url,  # Replace these with your URL and port
    port=server_config.port,  # Replace these with your URL and port
    verbose=True,
    num_workers=16,
    sleep_for_rate_limiting_sec=0.5,
    transaction_batch_size=250,
    start_height=start_height,
    end_height=end_height,
    # save_to_csv=output_file_path,
    save_to_feather=output_file_path.replace(".csv", ".feather"),
    run_on_init=True,
)
logger.info(f"\nDone!")
logger.info(f"Took {time.perf_counter() - tic:.2f} seconds to run over {end_height-start_height} blocks.")
logger.info(f"Output saved to {output_file_path}")
