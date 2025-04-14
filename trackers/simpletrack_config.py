def get_botsort_config():
    """Get optimized parameters for BoTSORT for dancing scenarios"""
    return {
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "match_thresh": 0.8,
        "track_buffer": 40,  # Increased for dancing scenarios
        "frame_rate": 30
    }

def get_bytetrack_config():
    """Get optimized parameters for ByteTrack for dancing scenarios"""
    return {
        "track_thresh": 0.5,
        "track_buffer": 40,  # Increased for dancing scenarios
        "match_thresh": 0.8,
        "frame_rate": 30
    }