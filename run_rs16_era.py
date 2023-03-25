from interface_logic import RunConfig
import server_config
from loguru import logger

logger.info("Starting ring size 16 era run...")

r = RunConfig(
    url=server_config.url,  # Replace these with your URL and port
    port=server_config.port,  # Replace these with your URL and port
    verbose=True,
    num_workers=16,
    start_height=2688888,
    end_height=None,  # 2683848  # < 1 weekish
    save_to_csv="/home/user/isthmus_dev/output/rs16_era.csv",
    run_on_init=True,
)

logger.info("\nDone!")
