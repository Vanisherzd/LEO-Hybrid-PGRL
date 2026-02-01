import numpy as np

def compute_ric(position_true, position_pred, velocity_true):
    """
    Compute Radial, Intrack, and Crosstrack errors.
    
    Args:
        position_true: (N, 3) array of true positions (km)
        position_pred: (N, 3) array of predicted positions (km)
        velocity_true: (N, 3) array of true velocities (km/s)
        
    Returns:
        err_ric: (N, 3) array where columns are [Radial, Intrack, Crosstrack] errors
    """
    
    # 1. Error Vector (Inertial Frame)
    err_eci = position_pred - position_true # (N, 3)
    
    # 2. Compute RIC Unit Vectors
    # Radial: Unit vector along Position
    r_norm = np.linalg.norm(position_true, axis=1, keepdims=True)
    u_r = position_true / r_norm
    
    # Crosstrack: Unit vector along Angular Momentum (r x v)
    h = np.cross(position_true, velocity_true)
    h_norm = np.linalg.norm(h, axis=1, keepdims=True)
    u_c = h / h_norm
    
    # Intrack: u_c x u_r (Along velocity, roughly)
    u_i = np.cross(u_c, u_r)
    
    # 3. Project Error onto Frames
    # Dot product row-wise
    e_r = np.sum(err_eci * u_r, axis=1)
    e_i = np.sum(err_eci * u_i, axis=1)
    e_c = np.sum(err_eci * u_c, axis=1)
    
    # Stack columns
    return np.stack([e_r, e_i, e_c], axis=1)

def compute_metrics(err_ric):
    """
    Compute aggregate metrics from RIC errors.
    """
    rmse = np.sqrt(np.mean(err_ric**2, axis=0))
    mae = np.mean(np.abs(err_ric), axis=0)
    max_err = np.max(np.abs(err_ric), axis=0)
    
    return {
        "RMSE_R": rmse[0], "RMSE_I": rmse[1], "RMSE_C": rmse[2],
        "Max_R": max_err[0], "Max_I": max_err[1], "Max_C": max_err[2]
    }
