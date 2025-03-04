import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.nn.functional as F
from utils import get_voxel_vertices  # Ensure you have this function

def expand_bits(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

def morton3D(x, y, z):
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)
    return xx | (yy << 1) | (zz << 2)


def morton3D_torch(grid_coordinates: torch.Tensor) -> torch.Tensor:
    x, y, z = grid_coordinates[:, 0], grid_coordinates[:, 1], grid_coordinates[:, 2]
    return torch.tensor([morton3D(int(a), int(b), int(c)) for a, b, c in zip(x, y, z)], dtype=torch.int32)


def morton3D_invert(x):
    x = x & 0x49249249
    x = (x | (x >> 2)) & 0xC30C30C3
    x = (x | (x >> 4)) & 0x0F00F00F
    x = (x | (x >> 8)) & 0xFF0000FF
    x = (x | (x >> 16)) & 0x0000FFFF
    return x


def morton3D_invert_torch(indices: torch.Tensor) -> torch.Tensor:
    coords = torch.zeros((indices.shape[0], 3), dtype=torch.int32)
    for i, ind in enumerate(indices):
        ind = int(ind)
        coords[i, 0] = morton3D_invert(ind >> 0)
        coords[i, 1] = morton3D_invert(ind >> 1)
        coords[i, 2] = morton3D_invert(ind >> 2)
    return coords


def packbits_pytorch(density_grid, density_threshold, density_bitfield):
    N = density_grid.size(0)

    for n in range(N):
        bits = 0
        for i in range(8):
            if density_grid[n, i] > density_threshold:
                bits |= (1 << i)
        density_bitfield[n] = bits

    return density_bitfield




 


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
    

# Helper function to calculate dt
def calc_dt(t, exp_step_factor, max_samples, grid_size, scale):
    # You can implement the logic based on your original CUDA function
    return exp_step_factor * (1 / (t + 1)) * (max_samples / grid_size) * scale

# Helper function to calculate mip from position
def mip_from_pos(x, y, z, cascades):
    # Calculate the mip level from position
    mx = torch.max(torch.abs(x), torch.max(torch.abs(y), torch.abs(z)))
    exponent = torch.floor(torch.log2(mx))  # Equivalent to frexpf behavior
    return torch.minimum(torch.tensor(cascades - 1), torch.maximum(torch.tensor(0), exponent + 1)).long()

# Helper function to calculate mip from dt
def mip_from_dt(dt, grid_size, cascades):
    # Calculate mip level from dt
    exponent = torch.floor(torch.log2(dt * grid_size))  # Equivalent to frexpf behavior
    return torch.minimum(torch.tensor(cascades - 1), torch.maximum(torch.tensor(0), exponent)).long()


class HashEmbedder(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bounding_box = torch.tensor([[0, 0, 0], [1, 1, 1]]).to(self.device)
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([
            nn.Embedding(2 ** self.log2_hashmap_size, self.n_features_per_level)
            for _ in range(n_levels)
        ])

        # Initialize embeddings
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)

        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]

        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]
        return c

    def forward(self, x):
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b ** i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
                x, self.bounding_box, resolution, self.log2_hashmap_size
            )
            hashed_voxel_indices = hashed_voxel_indices.to(self.device)
            self.embeddings[i] = self.embeddings[i].to(self.device)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        x_embedded_final = torch.cat(x_embedded_all, dim=-1)

        # Ensure output matches the expected shape (B, 16)
        if x_embedded_final.shape[-1] != 16:
            x_embedded_final = F.pad(x_embedded_final, (0, 16 - x_embedded_final.shape[-1]))

        return x_embedded_final