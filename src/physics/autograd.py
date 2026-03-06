import torch

def compute_time_derivatives(
    t: torch.Tensor, 
    moment_prediction: torch.Tensor
) -> torch.Tensor:
    """
    Computes the partial derivative of a predicted moment with respect to time (t).
    
    Args:
        t: The time tensor (must have requires_grad=True)
        moment_prediction: The output from MomentMLP (e.g., nascent_mean)
        
    Returns:
        The exact analytical derivative (d_moment / dt)
    """
    # Create a vector of ones to match the shape of the prediction for the Jacobian-vector product
    grad_outputs = torch.ones_like(moment_prediction, device=t.device)
    
    # torch.autograd.grad traces the computation graph backward from the prediction to t
    derivative = torch.autograd.grad(
        outputs=moment_prediction,
        inputs=t,
        grad_outputs=grad_outputs,
        create_graph=True,      # Crucial: allows us to backpropagate THROUGH the derivative
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return derivative
