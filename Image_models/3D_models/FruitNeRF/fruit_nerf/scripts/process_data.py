from typing import Union
import tyro
from typing_extensions import Annotated
from fruit_nerf.fruit_nerf_dataset import FruitNerfDataset

Commands = Union[Annotated[FruitNerfDataset, tyro.conf.subcommand(name="fruit")]]

def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()

if __name__ == "__main__":
    entrypoint()

def get_parser_fn():
    return tyro.extras.get_parser(Commands)