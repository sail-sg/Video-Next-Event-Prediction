import os
import tarfile
from huggingface_hub import snapshot_download

def download_dataset(data_folder: str) -> None:
    repo_id = "haonan3/V1-33K"
    dataset_dir = os.path.join(data_folder, "V1-33K")
    print(f"Downloading dataset: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_dir,
        cache_dir=dataset_dir
    )
    print("Download completed.")

def extract_archives(data_folder: str) -> None:
    dataset_dir = os.path.join(data_folder, "V1-33K")
    prefix_list = ["video_data_part", "first_segment_video_data_part"]
    start, end = 0, 70
    
    for prefix in prefix_list:
        print(f"Extracting archives in directory: {dataset_dir}")
        for i in range(start, end):
            tar_file = f"{prefix}{i}.tar.gz"
            tar_path = os.path.join(dataset_dir, tar_file)
            if not os.path.exists(tar_path):
                print(f"Archive not found: {tar_path}")
                continue
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=dataset_dir)
                print(f"Extracted: {tar_path}")
            except Exception as e:
                print(f"Failed to extract {tar_path}: {e}")
    print("Extraction completed.")

if __name__ == '__main__':
    data_folder = '.'  # Change this to your desired directory
    download_dataset(data_folder)
    extract_archives(data_folder)

