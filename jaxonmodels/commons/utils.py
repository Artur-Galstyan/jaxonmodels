import os
import pathlib
import re
import shutil
import urllib

import progressbar
from beartype.typing import Optional
from jaxtyping import PyTree
from loguru import logger


pbar = None
downloaded = 0
WEIGHTS_CACHE_DIR = os.path.expanduser("~/.cache/jaxonmodels")


def get_node(
    tree: PyTree, targets: list[str], log_when_not_found: bool = False
) -> PyTree | None:
    if len(targets) == 0 or tree is None:
        return tree
    else:
        next_target: str = targets[0]
        if bool(re.search(r"\[d\]", next_target)):
            split_index = next_target.rfind("[")
            name, index = next_target[:split_index], next_target[split_index:]
            index = index[1:-1]
            if hasattr(tree, name):
                subtree = getattr(tree, name)[int(index)]
            else:
                subtree = None
                if log_when_not_found:
                    logger.info(f"Couldn't find  {name} in {tree.__class__}")
        else:
            if hasattr(tree, next_target):
                subtree = getattr(tree, next_target)
            else:
                subtree = None
                if log_when_not_found:
                    logger.info(f"Couldn't find  {next_target} in {tree.__class__}")
        return get_node(subtree, targets[1:])


def pytorch_state_dict_str_to_pytree_str(string: str) -> str:
    res = ""
    parts = string.split(".")
    for part in parts:
        if part.isnumeric():
            part = f"[{part}]"
            res += part
        else:
            res += "." + part
    return res


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download_file(url: str, target_dir: Optional[str] = None) -> str:
    target_path = WEIGHTS_CACHE_DIR if target_dir is None else target_dir
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    target = pathlib.Path(target_path)
    file_name = url.split("/")[-1]

    if os.path.exists(target / file_name):
        logger.info(f"File {file_name} already exists in {target}")
        return str(target / file_name)

    logger.info(f"Downloading from {url}")
    urllib.request.urlretrieve(url, target / file_name, show_progress)  # pyright: ignore
    if os.path.exists(target / "__MACOSX"):
        shutil.rmtree(target / "__MACOSX")

    return str(target / file_name)
