from huggingface_hub import HfFileSystem
import gcsfs


hf_fs = HfFileSystem()
gcs_fs = gcsfs.GCSFileSystem()


with HfFileSystem().open() as hf_fs:
    with GCSFileSystem() as  
