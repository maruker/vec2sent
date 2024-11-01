import io
from typing import Any, Callable

import gdown
from platformdirs import user_data_dir
from os import getenv, mkdir, rmdir, listdir, remove
from os.path import join, exists
from zipfile import ZipFile
from logging import getLogger
from tqdm import tqdm
import json
import requests


def cleanup() -> None:
    """
    Deletes all models of sentence embeddings.
    Does a brief check for any subfolders that were not created by Vec2Sent itself.
    """
    contents = set(listdir(get_data_dir()))
    expected_contents = {"word_embeddings", "InferSent", "sent2vec", "quickthought"}
    if contents - expected_contents:
        raise RuntimeError("Unexpected files {} in cache folder {}. Not deleting cache out of caution."
                           .format(contents - expected_contents, get_data_dir()))
    rmdir(get_data_dir())


def get_data_dir() -> str:
    app_name = "Vec2Sent"
    author = "maruker"
    default_dir = user_data_dir(app_name, author, ensure_exists=True)
    return getenv("VEC2SENT_CACHE", default_dir)


def get_fasttext_embedding_path() -> str:
    word_embedding_path = join(get_data_dir(), "word_embeddings", "crawl-300d-2M.vec")
    if not exists(word_embedding_path):
        download_fasttext_embeddings()
    return word_embedding_path


def download_fasttext_embeddings() -> None:
    download_file_requests("word_embeddings", "crawl-300d-2M.vec.zip",
                           "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
    logger = getLogger(__name__)
    logger.info("Extracting word embeddings ...")
    zip_path = join(get_data_dir(), "word_embeddings", "crawl-300d-2M.vec.zip")
    zipped = ZipFile(zip_path, "r")
    zipped.extractall(join(get_data_dir(), "word_embeddings"))
    remove(zip_path)


def get_infersent_model_path() -> str:
    model_path = join(get_data_dir(), "InferSent", "infersent2.pkl")
    if not exists(model_path):
        download_infersent_data()
    return model_path


def download_infersent_data() -> None:
    download_file_requests("InferSent", "infersent2.pkl", "https://dl.fbaipublicfiles.com/infersent/infersent2.pkl")


def get_sent2vec_model_path() -> str:
    model_path = join(get_data_dir(), "sent2vec", "wiki_bigrams.bin")
    if not exists(model_path):
        download_sent2vec_data()
    return model_path


def download_sent2vec_data() -> None:
    download_file_requests("sent2vec", "wiki_bigrams.bin",
                           "https://drive.usercontent.google.com/download?id=0B6VhzidiLvjSaER5YkJUdWdPWU0&export=download&resourcekey=0-MVSyokxog2m4EQ4AGsssww&confirm=t&at=AN_67v2sXRISDyPOfdVv06LXnZGs%3A1730282976671")


def get_quickthought_model_path() -> str:
    model_path_1 = join(get_data_dir(), "quickthought", "BS400-W300-S1200-UMBC-bidir")
    model_path_2 = join(get_data_dir(), "quickthought", "BS400-W620-S1200-Glove-UMBC-bidir")
    if not exists(model_path_1) or not exists(model_path_2):
        download_quickthought_data()
    return join(get_data_dir(), "quickthought")


def download_quickthought_model(url: str) -> None:
    dl_buffer = io.BytesIO()
    download_into_buffer(url, dl_buffer)
    response_json = dl_buffer.getvalue().decode("utf-8")
    download_url = json.loads(response_json)["download_url"].replace('\\\\', '')
    filename = "BS400-W300-S1200-UMBC-bidir.zip"
    download_file_requests("quickthought", filename, download_url)
    logger = getLogger(__name__)
    logger.info("Extracting quickthought model ...")
    extract_path = join(get_data_dir(), "quickthought", filename.split(".zip")[0])
    mkdir(extract_path)
    zipped = ZipFile(join(get_data_dir(), "quickthought", filename))
    zipped.extractall(extract_path)
    remove(join(get_data_dir(), "quickthought", filename))


def download_quickthought_data() -> None:
    download_quickthought_model(
        "https://umich.app.box.cyiom/index.php?folder_id=48243154404&q%5Bshared_item%5D%5Bshared_name%5D=gwd0rp74lh6ogf6n69aqkhamab82tu8r&rm=box_v2_zip_shared_folder")
    download_quickthought_model(
        "https://umich.app.box.com/index.php?folder_id=48243576009&q%5Bshared_item%5D%5Bshared_name%5D=gwd0rp74lh6ogf6n69aqkhamab82tu8r&rm=box_v2_zip_shared_folder")


def get_glove_embedding_path() -> None:
    embedding_path = join(get_data_dir(), "word_embeddings", "glove.840B.300d.txt")
    if not exists(embedding_path):
        download_glove_embeddings()
    return embedding_path


def download_glove_embeddings() -> None:
    download_file_requests("word_embeddings", "glove.840B.300d.zip", "http://nlp.stanford.edu/data/glove.840B.300d.zip")
    logger = getLogger(__name__)
    logger.info("Extracting word embeddings ...")
    zip_path = join(get_data_dir(), "word_embeddings", "glove.840B.300d.zip")
    zipped = ZipFile(zip_path, "r")
    zipped.extractall()
    remove(zip_path)


def download_file_requests(path: str, filename: str, url: str) -> None:
    def download_fn(absolute_path: str, url: str) -> None:
        with open(join(absolute_path, filename), "wb") as f:
            download_into_buffer(url, f, filename)

    download_file(path, filename, url, download_fn)


def download_file_gdown(path: str, filename: str, url: str) -> None:
    def download_fn(absolute_path: str, url: str) -> None:
        gdown.download(url, output=join(absolute_path, filename))

    download_file(path, filename, url, download_fn)


def download_file(path: str, filename: str, url: str, download_fn: Callable[[str, str], None]) -> None:
    absolute_path = join(get_data_dir(), path)
    try:
        mkdir(absolute_path)
    except FileExistsError:
        pass

    if exists(join(absolute_path, filename)):
        return

    logger = getLogger(__name__)
    logger.info("Downloading to: " + absolute_path)

    download_fn(absolute_path, url)


def download_into_buffer(url: str, b: Any, desc: str = "download", chunk_size: int = 32 * 1024) -> None:
    """
    Download from url into buffer. Most useful for larger files.
    May throw an exception if the download fails.

    @param url: url to download
    @param desc: description to display in progress bar
    @param b: buffer to download into
    @param chunk_size: Amount of data to process at once
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_length = int(response.headers.get("Content-Length", 0))

        with tqdm(desc=desc, unit="B", unit_scale=True, unit_divisor=1024, total=total_length) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    b.write(chunk)
                    pbar.update(len(chunk))
