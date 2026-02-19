# import necessary libraries
import einops
import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from utils import MPRTransform

class RegNet(MPRTransform):
    """
    Registration Network that inherits from MPRTransform.
    Performs optimization of centerline transformations between different imaging modalities
    (e.g., CCTA and IVUS) through learnable parameters.
    """

    def __init__(self, data, config):
        """
        Initialize RegNet with data and configuration.

        Args:
            data: Data object containing CCTA, IVUS, MPR, and centerline information
            config: Configuration dictionary
        """
        self.c_m = config["MODEL"]

        # initialize parent class
        super(RegNet, self).__init__(
            config=config["DATA"]["mpr_transform"],
            device=f"cuda:{config['OPTIMIZATION']['gpu']}",
            max_points=data["CTL"]["R"]["ctl"].shape[0]
        )

        # set dimensions and spacing
        self.dim_ccta, self.spacing_ccta = data["CCTA"]["image"].shape, data["CCTA"]["spacing"]
        self.dim_ivus, self.spacing_ivus = data["IVUS"]["image"].shape, data["IVUS"]["spacing"]
        self.dim_mpr, self.spacing_mpr = data["MPR"]["image"].shape, data["MPR"]["spacing"]

        # process centerline
        self.centerline = self._process_centerline(data["CTL"]["J"].to(self.device))
        self.tangent, self.u_vec, self.v_vec = self.get_coordinate_frame(self.centerline, return_tangents=True)

        # initialize parameters
        self.identity = einops.repeat(torch.eye(3), "u v -> N u v", N=self.dim_ivus[2]).to(self.device)
        self._init_optimization_parameters()

    def _process_centerline(self, centerline: torch.Tensor) -> torch.Tensor:
        if self.config["resample"]:
            centerline = self.resample_centerline(centerline, resolution=self.spacing[2])

        # ensure centerline has the correct number of points
        return self.resample_centerline(centerline, size=self.dim_ivus[2])

    def _init_optimization_parameters(self):
        # radial scaling
        self.sr = torch.ones(1) * 0.8375
        if self.c_m["opt"]["r"]:
            self.sr = torch.nn.Parameter(self.sr)

        # z-axis transformation
        self.dz = torch.zeros(1)  # displacement
        self.sz = torch.ones(1)  # scaling
        if self.c_m["opt"]["z"]:
            self.dz = torch.nn.Parameter(self.dz)
            self.sz = torch.nn.Parameter(self.sz)

        # uv-plane displacement
        self.du = torch.zeros(self.dim_ivus[-1])
        self.dv = torch.zeros(self.dim_ivus[-1])
        if self.c_m["opt"]["uv"]:
            self.du = torch.nn.Parameter(self.du)
            self.dv = torch.nn.Parameter(self.dv)

        # rotation around centerline - CUMULATIVE formulation
        n_frames = self.dim_ivus[-1]
        n_ctrl_theta = self.c_m.get("n_ctrl_theta", 20)  # number of B-spline control points

        # global rotation offset (constant baseline rotation from initialization)
        self.theta_offset = torch.zeros(1)

        # relative twist vector (what we optimize) - all control points are free
        self.x_theta = torch.zeros(n_ctrl_theta)

        # precompute B-spline basis matrix for interpolation
        self.register_buffer('B_theta', self._compute_bspline_basis(n_frames, n_ctrl_theta))

        # non-rigid longitudinal deformation (B-spline arclength)
        n_ctrl_s = self.c_m.get("n_ctrl_s", 20)
        self.x_s = torch.zeros(n_ctrl_s)
        self.s_clamp_frac = self.c_m.get("s_clamp_frac", 0.35)
        self.register_buffer('B_s', self._compute_bspline_basis(self.dim_ivus[-1], n_ctrl_s))

        self.v_flip = torch.ones(1)
        self.x_s = self.x_s.to(self.device)
        if self.c_m["opt"]["theta"]:
            self.x_theta = torch.nn.Parameter(self.x_theta)

    def compute_cross_product_matrix(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-product matrix for each vector. Used in Rodrigues' rotation formula.
        """
        N = vectors.shape[0]
        A = torch.zeros(N, 3, 3).to(self.device)

        # fill the skew-symmetric matrix
        A[:, 0, 1] = -vectors[:, 2]
        A[:, 0, 2] = vectors[:, 1]
        A[:, 1, 0] = vectors[:, 2]
        A[:, 1, 2] = -vectors[:, 0]
        A[:, 2, 0] = -vectors[:, 1]
        A[:, 2, 1] = vectors[:, 0]

        return A

    def _compute_bspline_basis(self, n_points: int, n_ctrl: int, degree: int = 3) -> torch.Tensor:
        """
        Compute cubic B-spline basis matrix for interpolation.

        Args:
            n_points: Number of evaluation points (frames)
            n_ctrl: Number of control points
            degree: B-spline degree (default: 3 for cubic)

        Returns:
            Basis matrix of shape [n_points, n_ctrl]
        """
        # uniform knot vector
        n_knots = n_ctrl + degree + 1
        knots = torch.linspace(0, 1, n_knots - 2 * degree)
        knots = torch.cat([torch.zeros(degree), knots, torch.ones(degree)])

        # evaluation points (uniformly distributed along pullback)
        t = torch.linspace(0, 1 - 1e-6, n_points)

        # compute basis functions using de Boor's algorithm
        B = torch.zeros(n_points, n_ctrl)
        for i in range(n_ctrl):
            B[:, i] = self._bspline_basis_func(i, degree, knots, t)

        return B

    def _bspline_basis_func(self, i: int, k: int, knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Recursive B-spline basis function (de Boor's algorithm).

        Args:
            i: Control point index
            k: B-spline degree
            knots: Knot vector
            t: Evaluation points

        Returns:
            Basis function values at evaluation points
        """
        if k == 0:
            return ((knots[i] <= t) & (t < knots[i + 1])).float()

        B = torch.zeros_like(t)

        # first term
        denom1 = knots[i + k] - knots[i]
        if denom1 > 0:
            B += (t - knots[i]) / denom1 * self._bspline_basis_func(i, k - 1, knots, t)

        # second term
        denom2 = knots[i + k + 1] - knots[i + 1]
        if denom2 > 0:
            B += (knots[i + k + 1] - t) / denom2 * self._bspline_basis_func(i + 1, k - 1, knots, t)

        return B

    def _get_cumulative_theta(self) -> torch.Tensor:
        """
        Convert relative twist vector to cumulative rotation angles.

        Returns:
            Cumulative rotation angles for each frame [n_frames]
        """
        # cumulative sum of relative twists for control points
        delta_p = torch.cumsum(self.x_theta, dim=0)

        # apply B-spline interpolation to get smooth per-frame drift
        theta_drift = self.B_theta.to(self.x_theta.device) @ delta_p

        # add global rotation offset to cumulative drift
        theta = self.theta_offset.to(self.x_theta.device) + theta_drift

        return theta

    def _get_arclength_vector(self) -> torch.Tensor:
        """
        Compute per-frame arclength positions combining global affine (sz, dz)
        with non-rigid B-spline residual.

        Returns:
            Arclength positions in [-1, 1] for each frame [n_frames]
        """
        z_grid = torch.linspace(-1, 1, self.dim_ivus[-1], device=self.device)

        # global affine: inverse of (sz, dz) transform
        sz = self.sz.to(self.device)
        dz = self.dz.to(self.device)
        s_affine = (z_grid - dz) / sz

        # non-rigid B-spline residual with cumulative formulation
        n_ctrl_s = self.x_s.shape[0]
        inter_cp = 1.0 / max(n_ctrl_s - 1, 1)
        x_clamped = torch.clamp(self.x_s, -self.s_clamp_frac * inter_cp,
                                self.s_clamp_frac * inter_cp)

        # cumulative sum
        delta_p = torch.cumsum(x_clamped, dim=0)
        s_residual = self.B_s.to(self.x_s.device) @ delta_p

        return s_affine + s_residual

    def _get_arclength_residual(self) -> torch.Tensor:
        """
        Get just the B-spline residual component of the arclength deformation.
        Used for regularization.

        Returns:
            B-spline residual for each frame [n_frames]
        """
        n_ctrl_s = self.x_s.shape[0]
        inter_cp = 1.0 / max(n_ctrl_s - 1, 1)
        x_clamped = torch.clamp(self.x_s, -self.s_clamp_frac * inter_cp,
                                self.s_clamp_frac * inter_cp)

        delta_p = torch.cumsum(x_clamped, dim=0)
        return self.B_s.to(self.x_s.device) @ delta_p

    @property
    def theta(self) -> torch.Tensor:
        """
        Property to maintain compatibility with existing code.
        Returns cumulative rotation angles computed from relative twist vector.
        """
        return self._get_cumulative_theta()

    def set_constant_rotation(self, theta_val: float):
        """
        Set the global rotation offset to produce a constant baseline rotation.

        Args:
            theta_val: Desired constant baseline rotation value
        """
        with torch.no_grad():
            self.theta_offset.data.fill_(theta_val)
            self.x_theta.data.zero_()

    def compute_transformation_matrix(self, apply_displacement: bool = True) -> torch.Tensor:
        """
        Compute the transformation matrix for each point along the centerline.
        """
        # to radians
        theta_rad = self.theta * self.c_m["multiplier"] * math.pi

        # displacement transformation
        T_displacement = torch.eye(4).repeat(self.dim_ivus[-1], 1, 1).to(self.device)
        if apply_displacement:
            T_displacement[:, 0, 3] = self.du
            T_displacement[:, 1, 3] = self.dv
        T_displacement[:, 2, 3] = self.dz

        # scaling transformation
        T_scaling = torch.eye(4).repeat(self.dim_ivus[-1], 1, 1).to(self.device)
        T_scaling[:, 0, 0] = self.sr
        T_scaling[:, 1, 1] = self.sr
        T_scaling[:, 2, 2] = self.sz

        # rotation transformation
        T_rotation = torch.eye(4).repeat(self.dim_ivus[-1], 1, 1).to(self.device)
        T_rotation[:, 0, 0] = torch.cos(theta_rad)
        T_rotation[:, 0, 1] = -torch.sin(theta_rad)
        T_rotation[:, 1, 0] = torch.sin(theta_rad)
        T_rotation[:, 1, 1] = torch.cos(theta_rad)

        return T_rotation @ T_scaling @ T_displacement

    def resample_along_z(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Resample vectors along the z-axis using global affine + non-rigid
        B-spline arclength deformation.
        """
        s = self._get_arclength_vector()
        grid = self._create_z_sampling_grid()

        s_expanded = s[:, None].expand(-1, grid.shape[1])
        grid = torch.stack([grid[..., 0], s_expanded], dim=-1)

        return F.grid_sample(vectors[None, None], grid[None], align_corners=True, padding_mode='border')[0, 0]

    def return_z_grid(self):
        return self._get_arclength_vector()

    def _create_z_sampling_grid(self) -> torch.Tensor:
        """Create a grid for sampling along the z-axis."""
        l_steps = torch.linspace(-1, 1, self.dim_ivus[2])
        xyz_steps = torch.linspace(-1, 1, 3)

        l_grid, xyz_grid = torch.meshgrid(l_steps, xyz_steps, indexing='ij')
        grid = torch.stack([xyz_grid, l_grid], dim=-1)

        return grid.to(self.device)

    def rotate_frame_vectors(self,
                             t: torch.Tensor,
                             u: torch.Tensor,
                             v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate the frame vectors u and v around tangent t by cumulative theta angles.
        """
        # get cross-product matrices for Rodrigues' formula
        A = self.compute_cross_product_matrix(t)

        # get cumulative rotation angles
        theta_cumulative = self._get_cumulative_theta().to(self.device)
        theta_rad = theta_cumulative[:, None, None] * self.c_m["multiplier"] * math.pi

        # apply Rodrigues' formula: R = I + sin(theta)A + (1-cos(theta))A^2
        R = self.identity + torch.sin(theta_rad) * A + (1 - torch.cos(theta_rad)) * torch.matmul(A, A)

        # apply rotation to vectors
        u_rotated = torch.einsum('bij,bj->bi', R, u)
        v_rotated = torch.einsum('bij,bj->bi', R, v) * self.v_flip.to(self.device)

        return u_rotated, v_rotated

    def create_sampling_grid(self,
                             ctl: torch.Tensor,
                             t: torch.Tensor,
                             u: torch.Tensor,
                             v: torch.Tensor,
                             mode: str = "default") -> torch.Tensor:
        """
        Create a sampling grid based on the centerline and frame vectors.

        Args:
            ctl: Centerline points [N, 3]
            t, u, v: Frame vectors [N, 3]
            mode: "default" for Cartesian or "polar" for polar coordinates
        """
        # calculate displacement in UV plane
        du = self.du * self.margin[0] * self.sr.to(self.device)
        dv = self.dv * self.margin[1] * self.sr.to(self.device)
        d = u * du[:, None] + v * dv[:, None]

        ctl_displaced = ctl + d * self.spacing_ivus[0]

        if mode == "default":
            grid = self.cartesian_grid[..., :ctl.shape[0]]

            # calculate sampling coordinates
            xs = ctl_displaced[:, 0] + \
                 (grid[0] - self.margin[0]) * self.sr.to(self.device) * u[:, 0] * self.spacing_ivus[0] + \
                 (grid[1] - self.margin[1]) * self.sr.to(self.device) * v[:, 0] * self.spacing_ivus[1]

            ys = ctl_displaced[:, 1] + \
                 (grid[0] - self.margin[0]) * self.sr.to(self.device) * u[:, 1] * self.spacing_ivus[0] + \
                 (grid[1] - self.margin[1]) * self.sr.to(self.device) * v[:, 1] * self.spacing_ivus[1]

            zs = ctl_displaced[:, 2] + \
                 (grid[0] - self.margin[0]) * self.sr.to(self.device) * u[:, 2] * self.spacing_ivus[0] + \
                 (grid[1] - self.margin[1]) * self.sr.to(self.device) * v[:, 2] * self.spacing_ivus[1]

        else:  # polar mode
            angles = self._generate_angles(self.patch_size_polar[1], repeat=t.shape[0]) * self.v_flip.to(self.device)
            rotated_vectors = self._rotate_vectors_around_axis(t, u, angles)

            # generate polar grid
            grid = self.polar_grid[..., :ctl.shape[0]]
            radial_grid = grid[0] * self.sr.to(self.device)

            # calculate sampling coordinates
            xs = ctl_displaced[:, 0] + radial_grid * rotated_vectors[:, 0].T * self.spacing_polar[0]
            ys = ctl_displaced[:, 1] + radial_grid * rotated_vectors[:, 1].T * self.spacing_polar[0]
            zs = ctl_displaced[:, 2] + radial_grid * rotated_vectors[:, 2].T * self.spacing_polar[0]

        # normalize coordinates to [-1, 1] for grid_sample
        xs_norm = (xs / self.spacing_ccta[0]) / self.dim_ccta[0] * 2 - 1
        ys_norm = (ys / self.spacing_ccta[1]) / self.dim_ccta[1] * 2 - 1
        zs_norm = (zs / self.spacing_ccta[2]) / self.dim_ccta[2] * 2 - 1
        return torch.stack((zs_norm, ys_norm, xs_norm), dim=-1)[None]

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input MPR image using learned parameters to align with IVUS.
        """
        # ensure correct device and dimensions
        x = x.to(self.device)
        while len(x.shape) < 5:
            x = x.unsqueeze(0)

        T = torch.eye(4).to(self.device)
        T[2, 2] = self.sz
        T[2, 3] = self.dz
        T = torch.inverse(T)

        if self.v_flip < 0:
            # flip along x- and y-axis
            x = torch.flip(x, [2, 3])

        # apply z-transformation
        x = einops.rearrange(x, "b c x y z -> b c z y x")
        shape = (x.shape[0], x.shape[1], self.dim_ivus[-1], x.shape[3], x.shape[4])
        grid = F.affine_grid(T[None, :3], shape, align_corners=False).to(self.device)
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)

        # displacement
        T_d = torch.eye(4).repeat(self.dim_ivus[-1], 1, 1).to(self.device)
        T_d[:, 0, 3] = self.du
        T_d[:, 1, 3] = self.dv

        # scaling
        T_s = torch.eye(4).repeat(self.dim_ivus[-1], 1, 1).to(self.device)
        T_s[:, 0, 0] = self.sr
        T_s[:, 1, 1] = self.sr

        # rotation
        theta = self.theta * self.c_m["multiplier"] * math.pi - 0.5 * math.pi
        T_theta = torch.eye(4).repeat(self.dim_ivus[-1], 1, 1).to(self.device)
        T_theta[:, 0, 0] = torch.cos(theta)
        T_theta[:, 0, 1] = -torch.sin(theta)
        T_theta[:, 1, 0] = torch.sin(theta)
        T_theta[:, 1, 1] = torch.cos(theta)
        T = T_theta @ T_s @ T_d

        x = einops.rearrange(x, "b c z y x -> z b c y x")
        grid = F.affine_grid(T[:, :3], x.size(), align_corners=False).to(self.device)
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)

        return einops.rearrange(x > 0.5, "z b c y x -> b c x y z")

    def forward(self, image: torch.Tensor, mode: str = "default") -> torch.Tensor:
        """
        Main forward function that applies the MPR transformation with optimized parameters.
        """
        # first, prepare the image
        image = image.to(self.device)
        while len(image.shape) < 5:
            image = image.unsqueeze(0)

        # resample centerline and frame vectors with current parameters
        ctl = self.resample_along_z(self.centerline)
        t = self.resample_along_z(self.tangent)
        u = self.resample_along_z(self.u_vec)
        v = self.resample_along_z(self.v_vec)

        # rotate frame vectors
        u, v = self.rotate_frame_vectors(t, u, v)

        # create sampling grid and sample from image
        grid = self.create_sampling_grid(ctl, t, u, v, mode)

        x = F.grid_sample(image, grid, align_corners=True)
        return torch.flip(x, [2]) if mode == "polar" else x

    def get_current_frame(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the current centerline and frame vectors with all transformations applied.
        """
        # resample and transform centerline and frame vectors
        ctl = self.resample_along_z(self.centerline)
        t = self.resample_along_z(self.tangent)
        u = self.resample_along_z(self.u_vec)
        v = self.resample_along_z(self.v_vec)

        # rotate frame vectors
        u, v = self.rotate_frame_vectors(t, u, v)

        return ctl, u, v
