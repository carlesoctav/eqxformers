from huggingface_hub import HfFileSystem
import gcsfs
from tqdm import tqdm


def copy_hf_to_gcs(hf_path: str, gcs_path: str, token: str = None, chunk_size: int = 10 * 1024 * 1024):
    """
    Copy files from Hugging Face Hub to Google Cloud Storage without downloading locally.
    
    Args:
        hf_path: HuggingFace path (e.g., "datasets/my-username/my-dataset-repo/data")
        gcs_path: GCS path (e.g., "gs://bucket-name/path")
        token: Optional HuggingFace token for private repos
        chunk_size: Size of chunks to stream (default 10MB)
    """
    # Initialize filesystems
    hf_fs = HfFileSystem(token=token)
    gcs_fs = gcsfs.GCSFileSystem()
    
    # Ensure hf_path has the correct prefix
    if not hf_path.startswith("hf://"):
        hf_path = f"hf://{hf_path}"
    
    # Get all files from HF repo
    print(f"Listing files in {hf_path}...")
    all_files = hf_fs.glob(f"{hf_path}/**")
    
    # Filter out directories, keep only files
    files_to_copy = [f for f in all_files if hf_fs.isfile(f)]
    
    print(f"Found {len(files_to_copy)} files to copy")
    
    # Copy each file
    for hf_file in tqdm(files_to_copy, desc="Copying files"):
        # Get relative path
        relative_path = hf_file.replace(hf_path.replace("hf://", ""), "").lstrip("/")
        gcs_file = f"{gcs_path}/{relative_path}"
        
        print(f"Copying {hf_file} -> {gcs_file}")
        
        # Stream copy in chunks to avoid memory issues
        with hf_fs.open(hf_file, "rb") as src:
            with gcs_fs.open(gcs_file, "wb") as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
    
    print(f"Successfully copied {len(files_to_copy)} files to {gcs_path}")


if __name__ == "__main__":
    # Copy FineWeb sample-100BT dataset from HF to GCS
    hf_dataset_path = "datasets/HuggingFaceFW/fineweb/sample-100BT"
    gcs_destination = "gs://carles-git-good/100BT"
    
    print("Starting copy from HuggingFace to GCS...")
    copy_hf_to_gcs(hf_dataset_path, gcs_destination)
