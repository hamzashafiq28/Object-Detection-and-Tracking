def get_deepocsort_config():
    """Get optimized parameters for DeepOCSORT for dancing scenarios"""
    return {
        "det_thresh": 0.4,  # Slightly higher detection threshold for stability
        "max_age": 90,      # Keep tracks alive longer (increased from 30)
        "min_hits": 2,      # Lower this to recover tracks faster (from 3)
        "iou_threshold": 0.2,  # Lower for looser spatial association
        "delta_t": 5,       # Increased for better camera motion compensation
        "asso_func": "giou",  # Better than IoU for dancing poses
        "inertia": 0.1,     # Lower inertia for faster adaptation to direction changes
        "w_association_emb": 0.75,  # Higher weight on appearance (from 0.5)
        "alpha_fixed_emb": 0.9,     # Slightly lower to adapt appearance faster
        "aw_param": 0.3,    # Lower for better adaptive weighting
        "embedding_off": False,  # Keep embeddings ON
        "cmc_off": False,   # Keep camera motion compensation ON
        "aw_off": False,    # Keep adaptive weights ON
        "Q_xy_scaling": 0.02,  # Higher process noise for faster position updates
        "Q_s_scaling": 0.0005  # Higher process noise for scale updates   

    }