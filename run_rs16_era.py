start_height: int = 2688888
end_height: int = 2693928

from interface_logic import RunConfig
import server_config
from loguru import logger
from time import perf_counter


logger.info("Starting ring size 16 era run...")
tic: float = perf_counter()
r = RunConfig(
    url=server_config.url,  # Replace these with your URL and port
    port=server_config.port,  # Replace these with your URL and port
    verbose=True,
    num_workers=16,
    sleep_for_rate_limiting_sec=0.01,
    transaction_batch_size=250,
    start_height=start_height,
    end_height=end_height,
    save_to_csv="/home/user/isthmus_dev/output/rs16_era.csv",
    run_on_init=True,
)
logger.info(f"\nDone! Took {perf_counter() - tic:.2f} seconds to run over {end_height-start_height} blocks.")
