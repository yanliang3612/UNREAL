import torch

from models import SummerTrainer
from src.args import parse_args
from src.utils import set_random_seeds


def main() -> None:
    args = parse_args()
    set_random_seeds(0)
    torch.set_num_threads(2)

    trainer = SummerTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
