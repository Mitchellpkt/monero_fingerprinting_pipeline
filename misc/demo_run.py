from interface_logic import RunConfig
import server_config

test_filepath: str = "test_run_config.json"
r = RunConfig(
    url=server_config.url,  # Replace these with your URL and port
    port=server_config.port,  # Replace these with your URL and port
    verbose=True,
    num_workers=16,
    start_height=2000000,
    end_height=2000032,
    save_to_csv="/home/user/isthmus_dev/test_output.csv",
    run_on_init=True,
)
r.to_run_config_json_file(test_filepath)
