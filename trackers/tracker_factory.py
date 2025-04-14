from pathlib import Path
import torch

# Import BoxMOT trackers
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.boosttrack.boosttrack import BoostTrack
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

# Import configs
from .deepocsort_config import get_deepocsort_config
from .boosttrack_config import get_boosttrack_config
from .strongsort_config import get_strongsort_config
from .simpletrack_config import get_botsort_config, get_bytetrack_config

def create_tracker(tracker_type, device):
    """
    Create a tracker instance based on the specified type
    
    Args:
        tracker_type: Type of tracker (deepocsort, boosttrack, strongsort, botsort, bytetrack)
        device: Device to use (cuda or cpu)
        
    Returns:
        Initialized tracker
    """
    # Path to ReID model
    reid_weights = Path('REID_Weights/osnet_x0_25_msmt17.pt')
    reid_weights_exist = reid_weights.exists()
    
    # Convert device to appropriate format for different trackers
    device_param = 0 if str(device) == "cuda" else "cpu"
    fp16_param = True if str(device) == "cuda" else False
    
    try:
        if tracker_type == "deepocsort":
            # DeepOCSORT initialization
            config = get_deepocsort_config()
            
            if not reid_weights_exist:
                print("ReID model weights not found. Using default settings.")
                return DeepOcSort(
                    reid_weights=None,
                    device=device_param,
                    half=fp16_param,
                    **config
                )
            else:
                print("ReID model weights found. Using custom settings.")
                return DeepOcSort(
                    reid_weights=reid_weights,
                    device=0,
                    half=True,
                    **config
                )
                
                
        elif tracker_type == "boosttrack":
            # BoostTrack initialization
            config = get_boosttrack_config()
            
            if not reid_weights_exist:
                print("ReID model weights not found. Using default settings.")
                return BoostTrack(
                    device=device_param,
                    half=fp16_param
                )
            else:
                return BoostTrack(
                    reid_weights=reid_weights,
                    device=0,
                    half=True,
                    **config
                )
                
        elif tracker_type == "strongsort":
            # StrongSORT initialization
            config = get_strongsort_config()
            
            if not reid_weights_exist:
                print("ReID model weights not found. Using default settings.")
                return StrongSort(
                    reid_weights=None,
                    device=device_param,
                    half=fp16_param
                )
            else:
                return StrongSort(
                    reid_weights=reid_weights,
                    device=device_param,
                    half=fp16_param,
                    **config
                )
                
        elif tracker_type == "botsort":
            # BoTSORT initialization
            config = get_botsort_config()
            return BotSort(**config)
                
        elif tracker_type == "bytetrack":
            # ByteTrack initialization
            config = get_bytetrack_config()
            return ByteTrack(**config)
                
        else:
            print(f"Unknown tracker type: {tracker_type}. Using ByteTrack.")
            config = get_bytetrack_config()
            return ByteTrack(**config)
                
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Falling back to ByteTrack with default settings")
        return ByteTrack(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )