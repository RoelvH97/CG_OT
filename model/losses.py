# import necessary libraries
import math
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.modules.loss import _Loss

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=1, beta=1):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


def nmi_gauss(x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of similar shape"

    def gaussian_window(x, bins, sigma):
        assert x.ndim == 2, "Input tensor should be 2-dimensional."
        return torch.exp(
            -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
        ) / (math.sqrt(2 * math.pi) * sigma)

    x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
    x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
    p_XY = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
    p_XY = p_XY + e  # deal with numerical instability

    p_XY = p_XY / p_XY.sum((1, 2))[:, None, None]

    p_X = p_XY.sum(1)
    p_Y = p_XY.sum(2)

    I = (p_XY * torch.log(p_XY / (p_X[:, None] * p_Y[:, :, None]))).sum((1, 2))

    marg_ent_0 = (p_X * torch.log(p_X)).sum(1)
    marg_ent_1 = (p_Y * torch.log(p_Y)).sum(1)

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


def nmi_gauss_mask(x1, x2, x1_bins, x2_bins, mask, sigma=1e-3, e=1e-10):
    def gaussian_window_mask(x, bins, sigma):
        assert x.ndim == 1, "Input tensor should be 2-dimensional."
        return torch.exp(-((x[None, :] - bins[:, None]) ** 2) / (2 * sigma ** 2)) / (
                math.sqrt(2 * math.pi) * sigma
        )

    x1_windowed = gaussian_window_mask(torch.masked_select(x1, mask), x1_bins, sigma)
    x2_windowed = gaussian_window_mask(torch.masked_select(x2, mask), x2_bins, sigma)
    p_XY = torch.mm(x1_windowed, x2_windowed.transpose(0, 1))
    p_XY = p_XY + e  # deal with numerical instability

    p_XY = p_XY / p_XY.sum()

    p_X = p_XY.sum(0)
    p_Y = p_XY.sum(1)

    I = (p_XY * torch.log(p_XY / (p_X[None] * p_Y[:, None]))).sum()

    marg_ent_0 = (p_X * torch.log(p_X)).sum()
    marg_ent_1 = (p_Y * torch.log(p_Y)).sum()

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


class NMILoss(_Loss):
    def __init__(self, intensity_range=None, masked: bool = True,
                 n_bins: int = 64, sigma: float = 0.1, per_slice=True):
        super().__init__()
        self.intensity_range = intensity_range
        self.n_bins = n_bins
        self.per_slice = per_slice
        self.sigma = sigma
        self.use_mask = masked
        if masked:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.n_bins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.n_bins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss(fixed, warped, bins_fixed, bins_warped, sigma=self.sigma).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.n_bins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.n_bins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        if self.per_slice:
            loss = torch.zeros(fixed.shape[-1], device=fixed.device, dtype=torch.float32)

            for i in range(fixed.shape[-1]):
                loss[i] = -nmi_gauss_mask(fixed[..., i], warped[..., i], bins_fixed, bins_warped, mask[..., i], sigma=self.sigma)
            return loss
        else:
            return -nmi_gauss_mask(fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma)


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return ((2.0 / (tensor.numel() - 1.0))
                * (grad_output.detach() / (result.detach() * 2 + e))
                * (tensor.detach() - tensor.mean().detach()))


stablestd = StableStd.apply


def ncc(x1, x2, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc


def ncc_mask(x1, x2, mask, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    x1 = torch.masked_select(x1, mask)
    x2 = torch.masked_select(x2, mask)
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc


class NCCLoss(_Loss):
    def __init__(self, inverted=False, per_slice=False, masked=False, **kwargs):
        super().__init__()
        self.invert = inverted
        self.per_slice = per_slice
        if masked:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        factor = 1 if self.invert else -1
        if self.per_slice:
            # initialize loss vector of length shape[-1]
            loss = torch.zeros(fixed.shape[-1], device=fixed.device, dtype=torch.float32)

            # iterate over slices
            for i in range(fixed.shape[-1]):
                loss[i] = factor * ncc(fixed[..., i], warped[..., i])
            return loss
        else:
            return factor * ncc(fixed, warped)

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        factor = 1 if self.invert else -1
        if self.per_slice:
            loss = torch.zeros(fixed.shape[-1], device=fixed.device, dtype=torch.float32)

            for i in range(fixed.shape[-1]):
                loss[i] = factor * ncc_mask(fixed[..., i], warped[..., i], mask[..., i])
            return loss
        else:
            return factor * ncc_mask(fixed, warped, mask)


class SinkhornOTLoss(nn.Module):
    """
    Memory-efficient Unbalanced Sinkhorn OT loss for cylindrical probability maps.

    Uses convolutional Sinkhorn: O(N) memory instead of O(N^2) by exploiting
    translation-invariant cost structure via FFT-based convolutions.

    Designed for registration of probability maps:
    - Input: Batched probability maps [B, C, n_theta, n_z]
    - Handles differing probability masses via unbalanced OT (KL relaxation)
    - Respects cylindrical geometry (theta wraparound via circular convolution)
    - Fully differentiable and GPU-friendly

    Example usage:
        loss_fn = SinkhornOTLoss(epsilon=0.1, tau=0.8)
        loss = loss_fn(pred, target)  # [B, C, n_theta, n_z] tensors
    """

    def __init__(
            self,
            epsilon: float = 0.1,
            tau: float = 0.8,
            max_iter: int = 50,
            tol: float = 1e-4,
            z_weight: float = 1.0,
            reduction: str = 'mean',
            sigmoid: bool = True
    ):
        """
        Initialize Unbalanced Sinkhorn OT loss.

        Args:
            epsilon: Entropic regularization (blur radius in coordinate units).
                     Higher = faster/smoother, lower = more accurate OT.
                     Typical range: 0.05-0.5
            tau: Unbalanced OT parameter controlling marginal relaxation.
                 tau=1.0: Balanced OT (strict marginal constraints)
                 tau<1.0: Allows mass creation/destruction
                 Typical range: 0.5-0.95
            max_iter: Maximum Sinkhorn iterations.
            tol: Convergence tolerance for early stopping.
            z_weight: Weight for z-axis distance relative to theta.
            reduction: 'mean', 'sum', or 'none' over batch/channels.
        """
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.z_weight = z_weight
        self.reduction = reduction
        self.sigmoid = sigmoid

        # Cache for blur kernels (avoid recomputation)
        self._kernel_cache = {}

    def _get_blur_kernel(self, n_theta: int, n_z: int, device: torch.device) -> tuple:
        """
        Build Gaussian-like blur kernel for convolutional Sinkhorn.

        The kernel K(dx, dz) = exp(-C(dx, dz) / epsilon) where C is the
        cylindrical geodesic distance in NORMALIZED coordinates [0, 1].

        Returns:
            K_fft: FFT of Gibbs kernel for Sinkhorn iterations
            CK_fft: FFT of cost-weighted kernel for transport cost computation
        """
        cache_key = (n_theta, n_z, device)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # Coordinate offsets centered at (0, 0)
        theta_idx = torch.arange(n_theta, device=device, dtype=torch.float32)
        z_idx = torch.arange(n_z, device=device, dtype=torch.float32)

        # Circular distance on theta (geodesic on S^1), normalized to [0, 1]
        # Distance from position i to position 0, with wraparound
        # Max circular distance is n_theta/2 pixels
        d_theta_pixels = torch.minimum(theta_idx, n_theta - theta_idx)
        d_theta_norm = d_theta_pixels / (n_theta / 2)  # Normalized to [0, 1]

        # Linear distance on z, normalized to [0, 1]
        z_center = n_z // 2
        d_z_pixels = torch.abs(z_idx - z_center)
        d_z_norm = d_z_pixels / (n_z / 2)  # Normalized to [0, 1]

        # Build 2D distance kernel [n_theta, n_z]
        d_theta_2d = d_theta_norm.unsqueeze(1)  # [n_theta, 1]
        d_z_2d = d_z_norm.unsqueeze(0)  # [1, n_z]

        # Cylindrical geodesic distance in normalized coordinates
        # With z_weight=1.0, both axes contribute equally when at their max
        C = torch.sqrt(d_theta_2d ** 2 + (self.z_weight * d_z_2d) ** 2)

        # Gibbs kernel: K = exp(-C / epsilon)
        # NO normalization - Sinkhorn handles the scaling via dual potentials
        K = torch.exp(-C / self.epsilon)

        # Cost-weighted kernel for computing <C, P>: CK = C * K
        CK = C * K

        # Precompute FFT of kernels for fast convolution
        # Shift kernel so that center is at (0, 0) for proper convolution
        K_shifted = torch.roll(K, shifts=(0, -z_center), dims=(0, 1))
        K_fft = torch.fft.rfft2(K_shifted)

        CK_shifted = torch.roll(CK, shifts=(0, -z_center), dims=(0, 1))
        CK_fft = torch.fft.rfft2(CK_shifted)

        self._kernel_cache[cache_key] = (K_fft, CK_fft)
        return K_fft, CK_fft

    def _blur_conv(self, x: Tensor, K_fft: Tensor) -> Tensor:
        """
        Apply blur kernel via FFT convolution.

        Uses circular convolution in theta (wraparound) and
        zero-padded convolution in z.

        Args:
            x: [B, n_theta, n_z] input
            K_fft: Precomputed FFT of blur kernel

        Returns:
            Kx: [B, n_theta, n_z] blurred output
        """
        # FFT of input
        x_fft = torch.fft.rfft2(x)

        # Convolution in Fourier domain = pointwise multiplication
        conv_fft = x_fft * K_fft.unsqueeze(0)

        # Inverse FFT
        result = torch.fft.irfft2(conv_fft, s=x.shape[-2:])

        return result

    def _log_blur_conv(self, log_x: Tensor, K_fft: Tensor) -> Tensor:
        """
        Compute log(K @ exp(log_x)) = log(K @ x) in a numerically stable way.

        Uses the log-sum-exp trick by shifting values before exp().
        """
        # Shift by max to prevent overflow
        max_log_x = log_x.amax(dim=(1, 2), keepdim=True)
        shifted = log_x - max_log_x

        # Now exp(shifted) is safe
        exp_shifted = torch.exp(shifted)

        # Convolve
        conv_result = self._blur_conv(exp_shifted, K_fft)

        # Convert back to log-domain, adding back the shift
        log_result = torch.log(conv_result.clamp(min=1e-20)) + max_log_x

        return log_result

    def _sinkhorn_conv(
            self,
            a: Tensor,
            b: Tensor,
            K_fft: Tensor,
            CK_fft: Tensor
    ) -> Tensor:
        """
        Unbalanced Sinkhorn via convolutions (memory-efficient, numerically stable).

        Uses log-domain throughout to prevent overflow/underflow.
        Computes actual transport cost <C, P> using cost-weighted kernel.

        Args:
            a: [B, n_theta, n_z] source distribution
            b: [B, n_theta, n_z] target distribution
            K_fft: FFT of blur kernel
            CK_fft: FFT of cost-weighted kernel (C * K)

        Returns:
            cost: [B] OT cost per batch element (always non-negative)
        """
        # Log-domain representations
        log_a = torch.log(a.clamp(min=1e-20))
        log_b = torch.log(b.clamp(min=1e-20))

        # Initialize dual potentials (in log-scaled form: f/epsilon, g/epsilon)
        f_eps = torch.zeros_like(a)  # f / epsilon
        g_eps = torch.zeros_like(b)  # g / epsilon

        # Scaling factor for unbalanced OT
        if self.tau < 1.0:
            rho = self.epsilon * self.tau / (1.0 - self.tau)
            lambda_rho = rho / (rho + self.epsilon)
        else:
            lambda_rho = 1.0

        for _ in range(self.max_iter):
            f_eps_prev = f_eps.clone()

            # Update f: f/eps = lambda * (log_a - log(K @ exp(g/eps)))
            log_K_exp_g = self._log_blur_conv(g_eps, K_fft)
            f_eps = lambda_rho * (log_a - log_K_exp_g)

            # Update g: g/eps = lambda * (log_b - log(K @ exp(f/eps)))
            log_K_exp_f = self._log_blur_conv(f_eps, K_fft)
            g_eps = lambda_rho * (log_b - log_K_exp_f)

            # Clamp to prevent extreme values
            f_eps = f_eps.clamp(-100, 100)
            g_eps = g_eps.clamp(-100, 100)

            # Check relative convergence (more robust to scale)
            abs_diff = torch.max(torch.abs(f_eps - f_eps_prev))
            scale = 1.0 + torch.max(torch.abs(f_eps))
            rel_diff = abs_diff / scale
            if rel_diff < self.tol:
                break

        # Compute transport cost: <C, P> = <u, (C*K) @ v>
        # where u = exp(f/eps), v = exp(g/eps)
        # Use log-domain for stability
        u = torch.exp(f_eps.clamp(-50, 50))
        v = torch.exp(g_eps.clamp(-50, 50))

        # (C*K) @ v via convolution with cost-weighted kernel
        CK_v = self._blur_conv(v, CK_fft)

        # Transport cost = <u, CK_v> = sum(u * CK_v)
        cost = (u * CK_v).sum(dim=(1, 2))

        # Ensure non-negative (should be, but numerical safety)
        cost = cost.clamp(min=0)

        # Final safety check
        cost = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))

        return cost

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute unbalanced Sinkhorn OT loss between prediction and target.

        Args:
            pred: [B, C, n_theta, n_z] predicted probability maps
            target: [B, C, n_theta, n_z] target probability maps

        Returns:
            loss: Scalar loss (or [B, C] if reduction='none')
        """
        B, C, n_theta, n_z = pred.shape
        assert target.shape == pred.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Get blur kernels (cached)
        K_fft, CK_fft = self._get_blur_kernel(n_theta, n_z, pred.device)

        # Reshape: [B, C, n_theta, n_z] -> [B*C, n_theta, n_z]
        pred_flat = pred.reshape(B * C, n_theta, n_z)
        target_flat = target.reshape(B * C, n_theta, n_z)

        # Ensure non-negative
        pred_flat = torch.clamp(pred_flat, min=0)
        target_flat = torch.clamp(target_flat, min=0)

        # Normalize to probability distributions
        pred_mass = pred_flat.sum(dim=(1, 2), keepdim=True)
        target_mass = target_flat.sum(dim=(1, 2), keepdim=True)

        pred_norm = pred_flat / (pred_mass + 1e-10)
        target_norm = target_flat / (target_mass + 1e-10)

        # Compute OT cost
        ot_cost = self._sinkhorn_conv(pred_norm, target_norm, K_fft, CK_fft)

        # Scale by geometric mean of masses
        mass_scale = torch.sqrt(
            (pred_mass.squeeze() + 1e-10) * (target_mass.squeeze() + 1e-10)
        )
        ot_cost = ot_cost * mass_scale

        # Reshape back to [B, C]
        ot_cost = ot_cost.reshape(B, C)

        # Reduction
        if self.reduction == 'mean':
            return ot_cost.mean()
        elif self.reduction == 'sum':
            return ot_cost.sum()
        else:
            return ot_cost
