from middleware.coordinator import Coordinator
import argparse
from middleware.deepseek_local import DeepSeekCoder

if __name__ == "__main__":
    
    llm = DeepSeekCoder("/scratch/st-fhendija-1/nikta/deep_model")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/tr.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        '--restart', 
        default=False, 
        action=argparse.BooleanOptionalAction, 
        help="True: restart the process from the beginning. False: resume from the latest state."
    )
    args = parser.parse_args()

    cd = Coordinator(args.config, args.restart, llm=llm)
    cd._base_run()

