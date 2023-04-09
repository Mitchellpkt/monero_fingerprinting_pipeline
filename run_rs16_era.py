start_height: int = 2688888
end_height: int = 2860073

from interface_logic import RunConfig
import server_config
from loguru import logger
import time

logger.info("Starting ring size 16 era run...")
output_file_path: str = f"/home/user/isthmus_dev/output/rs16_era_{int(time.time())}.csv"
tic: float = time.perf_counter()
r = RunConfig(
    url=server_config.url,  # Replace these with your URL and port
    port=server_config.port,  # Replace these with your URL and port
    verbose=True,
    num_workers=16,
    sleep_for_rate_limiting_sec=0.01,
    transaction_batch_size=250,
    start_height=start_height,
    end_height=end_height,
    save_to_csv=output_file_path,
    run_on_init=True,
)
logger.info(f"\nDone!")
logger.info(f"Took {time.perf_counter() - tic:.2f} seconds to run over {end_height-start_height} blocks.")
logger.info(f"Output saved to {output_file_path}")
