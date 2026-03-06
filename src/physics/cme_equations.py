import torch
import torch.nn.functional as F

def compute_cme_residuals(
    # 1. Continuous NN Predictions (The State)
    nascent_mean: torch.Tensor, 
    mature_mean: torch.Tensor, 
    nascent_var: torch.Tensor, 
    mature_var: torch.Tensor, 
    cov_nm: torch.Tensor,
    
    # 2. Autograd Derivatives (The LHS of the ODEs)
    d_nascent_mean_dt: torch.Tensor, 
    d_mature_mean_dt: torch.Tensor, 
    d_nascent_var_dt: torch.Tensor, 
    d_mature_var_dt: torch.Tensor, 
    d_cov_nm_dt: torch.Tensor,
    
    # 3. Upstream Forcing & Physical Constants (The RHS parameters)
    a_t: torch.Tensor,     # Burst Frequency (from RNN)
    b_t: torch.Tensor,     # Burst Size (from RNN)
    beta: torch.Tensor,    # Splicing rate (Trainable scalar/vector)
    gamma: torch.Tensor    # Degradation rate (Trainable scalar/vector)
) -> torch.Tensor:
    """
    Computes the Mean Squared Error residuals for the 5-ODE CME system.
    All inputs must be PyTorch tensors with requires_grad=True.
    Shapes should ideally be (Batch_Collocation_Points, Num_Target_Genes).
    """
    
    # --- EQUATION 1 & 2: First Moments (Means) ---
    # The physical laws for average RNA abundance
    rhs_nascent_mean = (a_t * b_t) - (beta * nascent_mean)
    rhs_mature_mean = (beta * nascent_mean) - (gamma * mature_mean)
    
    # --- EQUATION 3, 4, & 5: Second Moments (Variances & Covariance) ---
    # Two-stage bursting CME (Shahrezaei & Swain 2008, Xu et al. 2016)
    #
    # dVar(N)/dt = a*b*(b+1) - 2*beta*Var(N)
    rhs_nascent_var = (a_t * b_t * (b_t + 1)) - (2 * beta * nascent_var)

    # dVar(M)/dt = gamma*<M> + 2*beta*Cov(N,M) - 2*gamma*Var(M)
    # The gamma*<M> term is the Poissonian noise from single-molecule degradation.
    # The 2*beta*Cov(N,M) term couples the upstream nascent noise into mature.
    rhs_mature_var = (gamma * mature_mean) + (2 * beta * cov_nm) - (2 * gamma * mature_var)

    # dCov(N,M)/dt = beta*Var(N) - (beta+gamma)*Cov(N,M)
    rhs_cov_nm = (beta * nascent_var) - ((beta + gamma) * cov_nm)
    
    # --- CALCULATE RESIDUALS (LHS vs RHS) ---
    # If the NN perfectly obeys physics, all of these MSEs will be exactly 0.0
    res_n_mean = F.mse_loss(d_nascent_mean_dt, rhs_nascent_mean)
    res_m_mean = F.mse_loss(d_mature_mean_dt, rhs_mature_mean)
    res_n_var  = F.mse_loss(d_nascent_var_dt, rhs_nascent_var)
    res_m_var  = F.mse_loss(d_mature_var_dt, rhs_mature_var)
    res_cov    = F.mse_loss(d_cov_nm_dt, rhs_cov_nm)
    
    # Return the unweighted sum of the physical residuals (L_phys)
    total_physics_loss = res_n_mean + res_m_mean + res_n_var + res_m_var + res_cov
    
    return total_physics_loss
