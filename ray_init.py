import ray
from glob import glob

#%%

data_store = "/mnt/d/data_store"

#os.listdir(data_store)

glob(f"{data_store}/*")

#%%

ray.init(
    # 1. KEEP THIS LOCAL: Uses native Linux filesystem for fast sockets/IPC
    _temp_dir="/tmp/ray",
    
    # 2. OFFLOAD THE HEAVY WEIGHTS: Large arrays spill to your D: drive
    _system_config={
        "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": ["/mnt/d/ray_spill"]}}'
    },
    
    # 3. CODE REPOSITORY: Points to your external storage directory
    runtime_env={
        "working_dir": "/mnt/d/data_store/distributed_work"
    }
)

