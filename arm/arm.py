import torch 
import numpy as np
from einops import rearrange
from kornia.utils.grid import create_meshgrid3d
from .utils_ import morton3D_torch, morton3D_invert_torch, packbits_pytorch
from .utils_ import HashEmbedder, TruncExp



class ARM:
    def __init__(self):
        self.grid_size = 128
        self.scale = 0.5
        self.cascades = max(1+int(np.ceil(np.log2(2 * self.scale))), 1)
        self.density_grid = torch.zeros(self.cascades, self.grid_size ** 3, device = "cuda")
        self.grid_coords = create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype = torch.int32).reshape(-1, 3)
        self.density_bitfield = torch.zeros(self.cascades * self.grid_size ** 3//8, dtype=torch.uint8, device="cuda")
        self.L = 16
        self.F = 2
        self.log2_T = 19
        self.N_min = 16
        self.NEAR_DISTANCE = 0.01
        self.MAX_SAMPLES = 1024
        self.b = np.exp(np.log(2048*self.scale/self.N_min)/(self.L-1))
        self.center = torch.zeros(1,3, device='cuda')
        self.xyz_min = -torch.ones(1,3, device = 'cuda')*self.scale
        self.xyz_max = torch.ones(1,3, device = 'cuda')*self.scale
        self.half_size = (self.xyz_max-self.xyz_min)/2
        self.T_threshold = 1e-4
        self.xyz_encoder = HashEmbedder(
            n_levels = self.L,
            n_features_per_level = self.F,
            log2_hashmap_size = self.log2_T,
            base_resolution = self.N_min,
            finest_resolution = self.b * self.N_min
        )
    
    def mark_invisible_cells(self, K, poses, img_wh, chunk=None):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        if chunk is None:
            chunk = self.MAX_SAMPLES * 32
        
        N_cams = poses.shape[0]
        N_cams = torch.tensor(N_cams)
        N_cams = N_cams.to('cuda')
        self.count_grid = torch.zeros_like(self.density_grid, device='cuda')
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3) rotation matrix
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1) translation matrix
        cells = self.get_all_cells()
        K = torch.tensor(K)
        #Determine whether a cell is visible to at least one camera
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w.to('cuda') + w2c_T # (N_cams, 3, chunk)
                uvd = K.float() @ xyzs_c.float() # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=self.NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                covered_by_cam = covered_by_cam.to('cuda')
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<self.NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)
    
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95):
        """
        Updates the density grid with occupancy values while preventing memory overflow.
        """
        with torch.no_grad():  # Prevents storing gradients for occupancy grid updates
            density_grid_tmp = torch.zeros_like(self.density_grid, dtype=torch.float16)

            # Determine which cells to update
            if warmup:
                cells = self.get_all_cells()
            else:
                cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4, density_threshold)

            chunk_size = self.MAX_SAMPLES * 32 # Process at most 64K (65536) cells per forward pass

            for c in range(self.cascades):
                indices, coords = cells[c]

                s = min(2**(c-1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)

                # Pick random position in the cell by adding noise in [-hgs, hgs]
                xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size

                # Process in smaller chunks to prevent OOM
                sigmas = []
                for i in range(0, xyzs_w.shape[0], chunk_size):
                    chunk_xyz = xyzs_w[i:i+chunk_size]
                    sigmas.append(self.density(chunk_xyz))

                sigmas = torch.cat(sigmas, dim=0).half()
                density_grid_tmp[c, indices] = sigmas

            # Apply decay factor, ensuring stability
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)

            # Update density grid: only update valid (non-negative) densities
            self.density_grid = torch.where(
                self.density_grid < 0,  # If already marked as invalid, keep it
                self.density_grid,
                torch.maximum(self.density_grid * decay, density_grid_tmp)  # Apply decay
            )

            # Compute mean density
            mean_density = self.density_grid[self.density_grid > 0].mean().item()

            # Pack updated density grid into bitfield
            self.density_bitfield = packbits_pytorch(
                self.density_grid, min(mean_density, density_threshold), self.density_bitfield
            )

    
    # def update_density_grid(self, density_threshold, warmup = False, decay = 0.95):
    #     """
    #     Updates the density grid 
    #     """
    #     density_grid_tmp = torch.zeros_like(self.density_grid, dtype = torch.float16)

    #     if warmup: # During first steps
    #         cells = self.get_all_cells()
    #     else:
    #         cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4, 
    #                                                        density_threshold=density_threshold)

    #     # Infer sigmas
    #     for c in range(self.cascades):
    #         indices, coords = cells[c]
    #         s = min(2**(c-1), self.scale)
    #         half_grid_size = s/self.grid_size
    #         xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)

    #         # pick random position in the cell by adding noise in [-hgs, hgs]
    #         xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
    #         # BIGGEST CHANGE W.R.T. ORIGINAL:
    #         #tmp= TruncExp.apply(xyzs_w[:, 0])
    #         #density_grid_tmp[c, indices] = tmp.half()
    #         density_grid_tmp[c, indices] = self.density(xyzs_w).half()

    #     decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)

    #     self.density_grid = torch.where(self.density_grid<0,
    #                         self.density_grid,
    #                         torch.maximum(self.density_grid*decay, density_grid_tmp))
        
    #     mean_density = self.density_grid[self.density_grid>0].mean().item()

    #     self.density_bitfield = packbits_pytorch(self.density_grid, 
    #                                              min(mean_density, density_threshold), 
    #                                              self.density_bitfield)
    
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                selected at each cascade
        
        """
        cells = []

        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M,3), dtype = torch.int32, device = 'cuda')
            indices1 = morton3D_torch(coords1).long()

            # Occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:,0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,), device = 'cuda')
                indices2 = indices2[rand_idx]

            coords2 = morton3D_invert_torch(indices2.int())

            # concatenate 
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells
    
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length of self.cascades) of indices and coords
                selected at each cascade
        """
        indices = morton3D_torch(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades
        return cells
   
    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs: 
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        x = x.to('cuda')
        
        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])

        return sigmas








