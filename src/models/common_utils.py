import urllib.request
import tqdm
import os


def download_with_progress(url: str, destination: str):
    """Download file with progress bar."""
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("content-length", 0))

    with tqdm.tqdm(
        total=total_size, unit="B", unit_scale=True, desc=os.path.basename(destination)
    ) as pbar:
        urllib.request.urlretrieve(
            url,
            destination,
            reporthook=lambda count, block_size, total_size: pbar.update(block_size),
        )