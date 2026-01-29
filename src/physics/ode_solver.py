import torch
from src.utils import J2

def analytical_j2_accel(state):
    """
    Computes J2 acceleration (normalized).
    state: (B, 6) or (B, 3) pos
    """
    x = state[:, 0:1]
    y = state[:, 1:2]
    z = state[:, 2:3]
    
    r = torch.sqrt(x**2 + y**2 + z**2)
    r2 = r**2
    r3 = r**3
    r5 = r**5
    
    factor_J2 = (1.5 * J2) / r5
    z2_r2 = (z / r)**2
    
    ax = - (x / r3) + factor_J2 * x * (5 * z2_r2 - 1)
    ay = - (y / r3) + factor_J2 * y * (5 * z2_r2 - 1)
    az = - (z / r3) + factor_J2 * z * (5 * z2_r2 - 3)
    
    return torch.cat([ax, ay, az], dim=1)

def ode_derivative(t, state, force_net):
    """
    Compute dy/dt = f(t, y)
    y = [r, v]
    dy/dt = [v, a_total]
    a_total = a_gravity_j2 + a_neural
    """
    position = state[:, 0:3]
    velocity = state[:, 3:6]
    
    # 1. Analytical Physics (J2)
    acc_analytical = analytical_j2_accel(position)
    
    # 2. Neural Physics (Perturbations: Drag, J3+, etc.)
    # The network takes the full state (r, v) to model drag (velocity dependent) 
    # and higher order gravity (position dependent).
    acc_neural = force_net(state)
    
    acc_total = acc_analytical + acc_neural
    
    # Derivative: [velocity, acceleration]
    return torch.cat([velocity, acc_total], dim=1)

def rk4_step(state, dt, force_net):
    """
    Performs one RK4 integration step.
    state: (B, 6)
    dt: scalar or (B, 1) step size
    force_net: nn.Module
    """
    k1 = ode_derivative(None, state, force_net)
    k2 = ode_derivative(None, state + 0.5 * dt * k1, force_net)
    k3 = ode_derivative(None, state + 0.5 * dt * k2, force_net)
    k4 = ode_derivative(None, state + dt * k3, force_net)
    
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

def integrate_trajectory(r0_v0, dt_seq, force_net, steps):
    """
    Integrates forward for 'steps'.
    dt_seq: scalar step size (normalized time). 
    """
    trajectory = [r0_v0]
    current_state = r0_v0
    
    for _ in range(steps):
        current_state = rk4_step(current_state, dt_seq, force_net)
        trajectory.append(current_state)
        
    # Stack: (Steps+1, Batch, 6) -> Permute to (Batch, Steps+1, 6)
    trajectory = torch.stack(trajectory, dim=0).permute(1, 0, 2)
    return trajectory
