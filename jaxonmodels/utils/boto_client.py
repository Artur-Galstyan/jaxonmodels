import os
import pathlib

import boto3
from beartype.typing import Any, Optional
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm


BUCKET_NAME = "jaxonmodels"
os.makedirs(os.path.expanduser("~/.jaxonmodels"), exist_ok=True)
DEFAULT_PATH = pathlib.Path(os.path.expanduser("~/.jaxonmodels"))


class BotoClient:
    @classmethod
    def get(cls):
        return boto3.client(
            service_name="s3",
            config=Config(
                signature_version=UNSIGNED,
                region_name="eu-central-1",
            ),
            endpoint_url="https://eu-central-1.linodeobjects.com",
        )


def download(object_key: str, target_path: Optional[str], s3: Any) -> pathlib.Path:
    meta_data = s3.head_object(Bucket=BUCKET_NAME, Key=object_key)
    path = (
        DEFAULT_PATH / object_key if target_path is None else pathlib.Path(target_path)
    )

    if path.exists():
        return path
    total_length = int(meta_data.get("ContentLength", 0))
    with tqdm(
        total=total_length,
        bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        with open(path, "wb") as f:
            s3.download_fileobj(BUCKET_NAME, object_key, f, Callback=pbar.update)
    return path
