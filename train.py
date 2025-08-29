import sys
import torch
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser


def main():
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help

    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]

    config = YamlParser(options["--config"]).get_config()

    if not cpu:
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Выводим информацию об ошибке и завершаем процесс кодом 1
        print(f"[FATAL] Unhandled exception: {e}", file=sys.stderr)
        sys.exit(1)
