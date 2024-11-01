import sys
from pathlib import Path
from typing import List

def add_to_path(path: List[str]) -> None:
    """
    This is unfortunately necessary because I am importing from a lot of old code that is not organized as a python module.
    A lot of old code uses absolute imports that assume the script is executed from the root folder of the repository.
    The only solution is to add the root folder to pythonpath.

    @param path: list of folders inside src/external containing the code (i.e. ["mos", "mos"] -> src/external/mos/mos)
    """
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.joinpath(*path)))
