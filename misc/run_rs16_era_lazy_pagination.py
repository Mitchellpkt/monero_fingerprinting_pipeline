start_height: int = 2688888
end_height: int = 2849982  # 2022-03-25
chunk_size: int = 5040  # 1 wk

from interface_logic import RunConfig
import server_config
from loguru import logger
import time
import numpy as np

height_range_iterables = [
    (x[0], x[-1])
    for x in np.array_split(
        range(start_height, end_height + 1), (end_height - start_height + 1) // chunk_size
    )
    if len(x)
]

logger.info(f"Starting ring size 16 era run in {len(height_range_iterables)} batches...")
tic: float = time.perf_counter()
for start_height, stop_height in height_range_iterables:
    output_file_path: str = f"/home/user/isthmus_dev/output/rs16_era_{start_height}_to_{stop_height}.csv"
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
    del r
    logger.info(f"Last output saved to saved to {output_file_path}")
logger.info(f"\nDone!")
logger.info(f"Took {time.perf_counter() - tic:.2f} seconds to run over {end_height-start_height} blocks.")
