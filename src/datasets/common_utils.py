import urllib.request
import tqdm
import os
import tarfile
import shutil


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


def extract_tarfile(tar_path: str, extract_path: str):
    """Extract tar file with progress bar."""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        with tqdm.tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_path)
                pbar.update(1)


def download_and_extract(
    url: str, download_dir: str, extract_dir: str, remove_tar: bool = True
):
    """Download and extract a tar file."""
    os.makedirs(download_dir, exist_ok=True)

    # Get filename from URL
    filename = os.path.basename(url).split("?")[0]  # Remove URL parameters
    tar_path = os.path.join(download_dir, filename)

    # Download if doesn't exist
    if not os.path.exists(tar_path):
        print(f"Downloading {filename}...")
        download_with_progress(url, tar_path)

    # Extract
    print(f"Extracting {filename}...")
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        with tqdm.tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_dir)
                pbar.update(1)

    # Clean up
    if remove_tar:
        os.remove(tar_path)
