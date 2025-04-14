def get_strongsort_config():
    """Get optimized parameters for StrongSORT for dancing scenarios"""
    return {
        "max_age": 60,
        "n_init": 2,
        "max_dist": 0.3,
        "use_ecc": False,  # Disable ECC to avoid convergence errors
        "ecc_lambda": 0.6,
        "nn_budget": 100,
        "mc_lambda": 0.995
    }