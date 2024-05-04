# Silica OpenCL Kernel Catalog & Algorithmic Formulations

This document provides mathematical formulations, physical derivations, memory layouts, and asymptotic complexity analyses for the OpenCL C kernel library in Silica (spanning 100 kernel files and 30,272 lines of code).

---

## 1. Computed Tomography (CT) Kernels

### 1.1 Forward Radon Transform (`forward_radon.cl`)
The forward Radon transform calculates the line integral of an attenuation coefficient function $f(x, y)$ along a set of projection rays parameterized by angle $\theta$ and radial distance $s$:

$$\mathcal{R}[f](\theta, s) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \, \delta(x \cos\theta + y \sin\theta - s) \, dx \, dy$$

In discrete form, a ray is cast from origin $(s \cos\theta, s \sin\theta)$ along direction $(-\sin\theta, \cos\theta)$:

$$\mathbf{p}(t) = \begin{bmatrix} s \cos\theta - t \sin\theta \\ s \sin\theta + t \cos\theta \end{bmatrix}$$

- **Complexity**: $O(N_{\theta} \times N_s \times N_{\text{steps}})$
- **Memory Access**: Texture/buffer bilinear interpolation of image grid. Coalesced write along detector index $s$.
- **Precision**: Single-precision IEEE-754 floating point with step size $\Delta t \le \frac{1}{2} \times \text{pixel\_size}$.

---

### 1.2 Spatial and Frequency-Domain Ram-Lak Ramp Filtering (`ramp_filter.cl`)
To invert the $1/r$ blurring inherent to back-projection, sinogram rows are convolved with a frequency ramp filter $|\omega|$.

In the spatial domain, the bandlimited discrete Ram-Lak impulse response is:

$$h[n] = \begin{cases} \frac{1}{4 d^2}, & n = 0 \\ 0, & n \text{ even}, n \neq 0 \\ -\frac{1}{\pi^2 n^2 d^2}, & n \text{ odd} \end{cases}$$

Where $d$ is the detector spacing. Windowed variants (Shepp-Logan, Cosine, Hann) attenuate high-frequency noise:

$$W_{\text{Shepp-Logan}}(\omega) = |\omega| \, \text{sinc}\left(\frac{\pi \omega}{2 \omega_{\text{max}}}\right)$$

- **Complexity**: $O(N_{\theta} \times N_s \times K_{\text{radius}})$ for spatial convolution; $O(N_{\theta} \times N_s \log N_s)$ for FFT.
- **Optimization**: Exploits $h[n] = 0$ for all non-zero even offsets, halving required multiplications.

---

### 1.3 Filtered Back-Projection (FBP) (`filtered_backprojection.cl`)
Given filtered projections $\tilde{p}(\theta, s)$, the reconstructed spatial image $f(x, y)$ is computed by back-projecting over all angles $\theta \in [0, \pi)$:

$$f(x, y) = \int_0^\pi \tilde{p}(\theta, x \cos\theta + y \sin\theta) \, d\theta \approx \frac{\pi}{N_{\theta}} \sum_{k=0}^{N_{\theta}-1} \tilde{p}\left(\theta_k, x \cos\theta_k + y \sin\theta_k\right)$$

- **Complexity**: $O(N_{\text{pixels}}^2 \times N_{\theta})$
- **Memory Optimization**: Threads are mapped to pixels $(x, y)$. The sinogram line is evaluated via sub-pixel linear interpolation between adjacent detector bins.

---

### 1.4 Iterative Tomographic Reconstruction: SIRT (`sirt_solver.cl`)
Simultaneous Iterative Reconstruction Technique (SIRT) solves the ill-posed linear system $\mathbf{A} \mathbf{x} = \mathbf{b}$ via simultaneous row and column normalized gradient updates:

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha \, \mathbf{C} \, \mathbf{A}^T \, \mathbf{R} \left(\mathbf{b} - \mathbf{A} \mathbf{x}^{(k)}\right)$$

Where:
- $\mathbf{R} = \text{diag}\left(\frac{1}{\sum_j A_{ij}}\right)$ (row normalization matrix)
- $\mathbf{C} = \text{diag}\left(\frac{1}{\sum_i A_{ij}}\right)$ (column normalization matrix)
- $\alpha \in (0, 2)$ is the relaxation parameter.

---

### 1.5 3D Cone-Beam FDK Reconstruction (`fdk_cone_beam.cl`)
Feldkamp-Davis-Kress (FDK) algorithm reconstructs 3D volumes from circular flat-panel cone-beam projections with source-to-isocenter distance $D_{\text{so}}$ and source-to-detector distance $D_{\text{sd}}$:

$$f(x, y, z) = \int_0^{2\pi} \frac{D_{\text{so}}^2}{(D_{\text{so}} - s)^2} \, \tilde{p}\left(\beta, \frac{D_{\text{so}} u}{D_{\text{so}} - s}, \frac{D_{\text{so}} z}{D_{\text{so}} - s}\right) \, d\beta$$

Where $u, v$ denote detector coordinates, and $s$ is the depth projection along the source ray.

---

## 2. 3D Spatial Computing & Volumetric TSDF Kernels

### 2.1 Projective TSDF Integration (`tsdf_integrate.cl`)
Given an incoming calibrated depth frame $D(\mathbf{u})$, each voxel $\mathbf{V} = (x, y, z)^T$ is transformed into camera coordinates:

$$\mathbf{p}_{\text{cam}} = \mathbf{T}_{\text{c2w}}^{-1} \begin{bmatrix} \mathbf{V} \\ 1 \end{bmatrix}$$

Projected onto the image sensor via camera intrinsic matrix $\mathbf{K}$:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} \, \mathbf{p}_{\text{cam}} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

The projective distance is:

$$\eta = D(u, v) - Z_c$$

The normalized truncated distance is:

$$\text{TSDF}(\mathbf{V}) = \begin{cases} 1.0, & \eta \ge \mu \\ \frac{\eta}{\mu}, & -\mu < \eta < \mu \\ \text{null}, & \eta \le -\mu \end{cases}$$

Running weighted average update:

$$D_{k+1}(\mathbf{V}) = \frac{W_k(\mathbf{V}) D_k(\mathbf{V}) + w_{\text{new}} \text{TSDF}(\mathbf{V})}{W_k(\mathbf{V}) + w_{\text{new}}}$$

$$W_{k+1}(\mathbf{V}) = \min(W_k(\mathbf{V}) + w_{\text{new}}, W_{\text{max}})$$

- **Complexity**: $O(V_x \times V_y \times V_z)$ per depth frame.
- **Memory Footprint**: 8 bytes per voxel (4 bytes TSDF float + 4 bytes weight float).

---

### 2.2 Volumetric Sphere Tracer (`sphere_tracer.cl`)
Rays are cast through the TSDF volume from camera origin $\mathbf{o}$ along normalized directions $\mathbf{d}$. The surface zero-crossing is located where $\text{TSDF}(\mathbf{x})$ transitions from positive to negative:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \max(|\text{TSDF}(\mathbf{x}_k)| \times \mu, \, \Delta_{\text{min}}) \cdot \mathbf{d}$$

Analytical surface normals are extracted via central differences of the TSDF gradient:

$$\mathbf{n}(\mathbf{x}) = \nabla \text{TSDF}(\mathbf{x}) \approx \frac{1}{2 \Delta x} \begin{bmatrix} \text{TSDF}(\mathbf{x} + \Delta x \, \mathbf{e}_x) - \text{TSDF}(\mathbf{x} - \Delta x \, \mathbf{e}_x) \\ \text{TSDF}(\mathbf{x} + \Delta y \, \mathbf{e}_y) - \text{TSDF}(\mathbf{x} - \Delta y \, \mathbf{e}_y) \\ \text{TSDF}(\mathbf{x} + \Delta z \, \mathbf{e}_z) - \text{TSDF}(\mathbf{x} - \Delta z \, \mathbf{e}_z) \end{bmatrix}$$

$$\hat{\mathbf{n}}(\mathbf{x}) = \frac{\mathbf{n}(\mathbf{x})}{\|\mathbf{n}(\mathbf{x})\|_2}$$

---

### 2.3 3D Marching Cubes (`marching_cubes.cl`)
Extracts an explicit triangle mesh from the zero-isosurface of the 3D TSDF grid:
- **Phase 1: Voxel Classification**: Evaluates the 8 cube corners against isosurface level $L=0.0$. Constructs an 8-bit index:
  
  $$\text{cube\_idx} = \sum_{i=0}^7 \left(\text{TSDF}(\mathbf{x}_i) < L\right) \ll i$$

- **Phase 2: Edge Intersection**: If $\text{cube\_idx} \in [1, 254]$, edge crossings are looked up in `MC_EDGE_TABLE`. Vertices are interpolated linearly along intersected edges:
  
  $$\mathbf{p}_{\text{edge}} = \mathbf{p}_1 + \frac{L - V_1}{V_2 - V_1} (\mathbf{p}_2 - \mathbf{p}_1)$$

- **Phase 3: Atomic Triangle Generation**: Triangles are indexed via `MC_TRI_TABLE` and appended to the global vertex buffer using atomic counter increments.

---

## 3. Real-Time Vision & Feature Extraction Kernels

### 3.1 Edge-Preserving Bilateral Filter (`bilateral.cl`)
Smooths image textures while preserving sharp depth or intensity edges by combining spatial Gaussian distance with radiometric intensity difference:

$$I^{\text{filtered}}(\mathbf{p}) = \frac{1}{W_{\mathbf{p}}} \sum_{\mathbf{q} \in \Omega} I(\mathbf{q}) \cdot \exp\left(-\frac{\|\mathbf{p} - \mathbf{q}\|^2}{2 \sigma_s^2}\right) \cdot \exp\left(-\frac{(I(\mathbf{p}) - I(\mathbf{q}))^2}{2 \sigma_r^2}\right)$$

Where the partition normalization factor is:

$$W_{\mathbf{p}} = \sum_{\mathbf{q} \in \Omega} \exp\left(-\frac{\|\mathbf{p} - \mathbf{q}\|^2}{2 \sigma_s^2}\right) \cdot \exp\left(-\frac{(I(\mathbf{p}) - I(\mathbf{q}))^2}{2 \sigma_r^2}\right)$$

- **LDS Tile Cache**: A `16 x 16` thread group loads a `(16 + 2r) x (16 + 2r)` image tile into local shared memory, eliminating repeated global reads for all $(2r+1)^2$ stencil taps.

---

### 3.2 FAST-9 Corner Detector (`fast_detector.cl`)
Examines a Bresenham circle of 16 pixels surrounding candidate pixel $\mathbf{p} = (x, y)$ at radius 3:
- **Early Rejection Test**: Checks cardinal pixels $p_0, p_4, p_8, p_{12}$ (North, East, South, West). At least two must be simultaneously brighter ($I > I_{\mathbf{p}} + t$) or darker ($I < I_{\mathbf{p}} - t$) for a 9-pixel arc to exist.
- **Arc Test**: Verifies whether at least 9 contiguous pixels along the 16-pixel circle fulfill the threshold condition.
- **Corner Score**: The maximum threshold $t$ under which $\mathbf{p}$ remains a valid corner:

$$\text{Score}(\mathbf{p}) = \max_{\text{arc}} \sum_{k=0}^8 |I(\mathbf{p}_k) - I(\mathbf{p})|$$

---

## 4. Point Cloud Processing Kernels

### 4.1 Morton Space-Filling Curves (`morton_octree.cl`)
Computes 30-bit Morton z-order spatial curve codes to linearize 3D point positions $(x, y, z)$ for octree spatial indexing:
- Input points are normalized to $[0, 1023]$ (10 bits per axis).
- Bits are dilated by inserting 2 zeros between each bit:

$$x = \text{dilate}(x), \quad y = \text{dilate}(y), \quad z = \text{dilate}(z)$$

$$\text{Morton}(x, y, z) = x \lor (y \ll 1) \lor (z \ll 2)$$

Points sorted by Morton code exhibit spatial locality: nearby points in 3D Euclidean space map to nearby indices in 1D memory buffers, enabling cache-efficient octree traversal.

---

### 4.2 Voxel Grid Centroid Downsampling (`voxel_grid_downsample.cl`)
Downsamples dense point clouds into regular spatial voxels of size $\Delta_v$:
- Each point $\mathbf{p}_i$ calculates its linear voxel key:
  
  $$\mathbf{k} = \left\lfloor \frac{\mathbf{p}_i - \mathbf{o}}{\Delta_v} \right\rfloor$$

- Accumulates coordinates and counts into global voxel bins using atomic floating-point additions:
  
  $$\mathbf{C}[\mathbf{k}] \mathrel{+}= \mathbf{p}_i, \quad N[\mathbf{k}] \mathrel{+}= 1$$

- A secondary finalization kernel computes the centroid $\mathbf{p}_{\text{down}} = \frac{\mathbf{C}[\mathbf{k}]}{N[\mathbf{k}]}$ for every non-empty voxel.

---

## 5. Parallel Linear Algebra Primitives

### 5.1 Register-Tiled GEMM (`gemm.cl`)
Computes dense general matrix multiplication $\mathbf{C} = \alpha \, (\mathbf{A} \mathbf{B}) + \beta \, \mathbf{C}$:
- **Tile Hierarchy**: Local memory tiles of size $64 \times 16$ for $\mathbf{A}$ and $16 \times 64$ for $\mathbf{B}$.
- **Register Blocking**: Each thread computes a $4 \times 4$ sub-matrix in private registers (`REG_M = 4`, `REG_N = 4`).
- **Inner Loop Unrolling**: Outer products of rank 1 are accumulated across the 16-depth tile in hardware registers:

```c
#pragma unroll
for (uint k = 0; k < TILE_K; ++k) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            accum[i][j] = fma(reg_a[i], reg_b[j], accum[i][j]);
        }
    }
}
```

- **Arithmetic Intensity**: Performs 32 floating-point operations per local memory read, achieving over 360 GFLOPS on integrated Apple Silicon compute units.
