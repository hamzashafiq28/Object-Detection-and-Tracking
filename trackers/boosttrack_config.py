def get_boosttrack_config():
    """Get optimized parameters for BoostTrack for dancing scenarios"""
    return {
        "max_age": 90,            # Increased from 60 to maintain tracks during longer occlusions
        "min_hits": 2,            # Decreased from 3 to recover tracks faster after reappearance
        "det_thresh": 0.7,        # Slightly lowered to detect partially occluded dancers
        "iou_threshold": 0.25,    # Lowered to accommodate varying poses during dancing
        "use_ecc": False,         # Disable ECC to avoid convergence errors
        "min_box_area": 10,       # Keep minimum area requirement
        "aspect_ratio_thresh": 2.0, # Increased to accommodate varied dance poses
        
        # BoostTrack parameters
        "lambda_iou": 0.4,        # Decreased IoU weight for dancing scenarios
        "lambda_mhd": 0.3,        # Increased Mahalanobis distance weight
        "lambda_shape": 0.3,      # Increased shape importance for dancers
        "use_dlo_boost": True,    # Keep deep learned occlusion boost
        "use_duo_boost": True,    # Keep detection uncertainty occlusion boost
        "dlo_boost_coef": 0.75,   # Increased from 0.65 for better occlusion handling
        "s_sim_corr": True,       # Enable shape similarity correlation for dance poses
        
        # BoostTrack++ parameters
        "use_rich_s": True,       # Enable rich shape features for better pose handling
        "use_sb": True,           # Enable spectral-based motion compensation
        "use_vt": True,           # Enable virtual tracklets for long occlusions
        
        "with_reid": True       
    }