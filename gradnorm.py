from torch import Tensor, nn
import torch
from files.torchjd.src.torchjd.scalarization._scalarizer_base import Scalarizer

class GradNormScalarizer(Scalarizer):
    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        # These are your "volume knobs"
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.alpha = alpha

        # This is your "Birth Name" storage
        self.register_buffer("initial_losses", None)

    def forward(self, values: Tensor) -> Tensor:
        if self.initial_losses is None:
            self.initial_losses = values.detach().clone()
        
        # Apply the weighted sum to the model's loss
        return (values * self.weights).sum()
    
    def _compute_gradient_norms(self, values: Tensor, model: nn.Module) -> Tensor:
        norms = []
        for loss in values:
            # Gradients of each task w.r.t. model parameters
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            # Calculate the L2 norm (magnitude) of the gradients
            norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
            norms.append(norm)
        return torch.stack(norms)

    def compute_gradnorm_loss(self, values: Tensor, model: nn.Module) -> Tensor:
        """This function balances the tasks."""
        # 1. Get current gradient norms
        norms = self._compute_gradient_norms(values, model)
        
        # 2. Calculate relative training rates (ratios)
        ratios = values / self.initial_losses
        
        # 3. Calculate target norms
        # Mean of gradients serves as the 'center'
        target = torch.mean(norms) * (ratios ** self.alpha)
        
        # 4. Balancing loss (how far off we are from the target)
        return torch.sum(torch.abs(norms - target.detach()))