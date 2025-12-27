"""
Name    : defaults.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : Central parameter configuration for atomic‑IQ.
"""

DEFAULTS = dict(
    opt_methods=["iq1","iq2"],  
    n_assets   = 10,             
    lookback   = 20,             
    cost_bps   = 10.0,            # turnover penalty (bps) in optimiser objective
    n_trials   = 5000,            # optuna trials (set to 1 for single random draw)
    seed       = 260370,         
    begin_date = "1987-12",       # "2000-01" or "2000-01-31"; None => full sample
    end_date   = "1999-12",       # None defaults to full sample
    part2_begin_date = None,      # None defaults to end_date + 1
    part2_end_date   = "2024-12",

    prices_csv = "./prices_multi_asset_master.csv",
    rf_csv     = "./DGS3MO_monthly_rf.csv",

    export_json = {
        "iq1": "aiq_params_iq1.json",
        "iq2": "aiq_params_iq2.json",
    },

    # seeding optuna 
    seed_identity_trial = True,   
    identity_logit_hi   =  4.0,    
    identity_logit_lo   = -4.0,  

    aiq_json_iq1 = "aiq_params_iq1.json", # storing optimised parameter set for iq1
    aiq_json_iq2 = "aiq_params_iq2.json", # as above for iq2

    rf_frequency      = "M",                 # "M" monthly (used only if interpretation == "annualized_yield")
    rf_align_method   = "ffill",             # {"ffill","bfill"}
    rf_interpretation = "monthly_effective", # {"monthly_effective","annualized_yield"}
    rf_column         = None,                # None => first column
   
    gamma_mode = "signed",   # {"signed","raw"}; sign selects the recency edge when "signed"
    u_gamma    = 0.10,       # upper bound for abs(gamma) 
    a_floor    = 0.00,       # small floor in temporal weights (0 disables)

    # channel weights & PSD
    PSD_check       = True,     # warns if matrix is not PSD
    weight_mode     = "free",   # {"free","simplex","basic_eta","clipped"}
    psd_repair      = "auto",   # {"auto","model_free","off"}
    target_min_eig  = None,     # impose a standing eigenfloor
    target_kappa    = None,     # cap matrix condition number
)

# scalars => fixed value; lists => categorical choices; tuples => numeric ranges [low, high].
DEFAULTS["param_space"]=dict(
   
    threshold     = [0.5],
    delta         = (1.0, 3.0),
    gamma         = None, # None -> (-u_gamma, +u_gamma) at runtime
    epsilon       = None, # None -> (0, lookback-1) at runtime
    center_method = ["mean", "median","zero"],
    scale_method  = ["vols", "pair_max", "pair_avg", "pair_min"],
)

# number of logits based on weight_mode
_mode = DEFAULTS["weight_mode"]
if _mode == "free":            # 'free' => [independent] sigmoid on each of [wB, wW, wT], and wI = 1-sum(wB,wW,wT)
    _L = 3
elif _mode == "simplex":       # 'simplex' => softmax on [wB, wW, wT, wI]
    _L = 4
elif _mode == "basic_eta":     # 'basic_eta' => sigmoid(eta) only on [wB], fixed eta^2 
    _L = 1                     # on [wW], fixed 1 on [wT], and wI = 1-sum(wB,wW,wT)
else:
    raise ValueError(f"Unknown weight_mode '{_mode}'")
DEFAULTS["param_space"]["logits"] = [(-4.0, 4.0)] * _L 
