# download_qwen_model.py

from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: str):
    """
    Download a full snapshot of the given Hugging Face repo.
    """
    print(f"Downloading {repo_id} into '{local_dir}' …")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("Download complete.")


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    LOCAL_DIR = "Qwen/Qwen2.5-0.5B-Instruct"
    download_model(MODEL_ID, LOCAL_DIR)
