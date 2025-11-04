# Assignment 2: Transformations & Filtering - Solution Documentation

## Part 1: Transformations in 3D

### Question 1.1: Reference Frame Definitions

In this exercise, we need to express key points in both world and camera coordinate systems.

**Given Information:**
- Distance d = 1.0
- The camera frame is rotated and translated relative to the world frame
- The camera's z-axis and x-axis lie in the plane defined by the world's z and x axes
- The camera's y-axis is parallel to the world's y-axis

**Solution:**

```python
w_wrt_world = np.array([0.0, 0.0, 0.0])  # World origin in world coordinates
w_wrt_camera = np.array([0.0, 0.0, d])    # World origin in camera coordinates
c_wrt_world = np.array([1.0/np.sqrt(2.0), 0.0, 1.0/np.sqrt(2.0)])  # Camera origin in world coordinates
c_wrt_camera = np.array([0.0, 0.0, 0.0])  # Camera origin in camera coordinates
```

**Explanation:**
- The world origin `w_wrt_camera` is at distance `d` along the camera's positive z-axis
- The camera origin `c_wrt_world` is located at $(\frac{d}{\sqrt{2}}, 0, \frac{d}{\sqrt{2}})$, which represents a 45° rotation in the xz-plane at distance d from the world origin
- By definition, each origin is at $(0, 0, 0)$ in its own coordinate system

---

### Question 1.2: World ⇨ Camera Transforms

This exercise requires deriving the homogeneous transformation matrix $T$ that converts points from world coordinates to camera coordinates.

**Transformation Matrix Structure:**

$$T = \begin{bmatrix}
R & t \\
\vec{0}^\top & 1
\end{bmatrix}$$

where $R \in \mathbb{R}^{3\times 3}$ is the rotation matrix and $t \in \mathbb{R}^{3\times 1}$ is the translation vector.

**Solution:**

```python
def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    T = np.eye(4)
    
    # Camera origin in world coordinates
    c_wrt_world = np.array([d/np.sqrt(2.0), 0.0, d/np.sqrt(2.0)])
    
    # Rotation: around y-axis by -135 degrees (or -3π/4 radians)
    theta = -3.0 * np.pi / 4.0
    R = np.array([
        [ np.cos(theta), 0.0, -np.sin(theta)],
        [ 0.0,           1.0,  0.0],
        [ np.sin(theta), 0.0,  np.cos(theta)]
    ])
    
    # Translation: t = -R × c_wrt_world
    t = - R @ c_wrt_world
    
    T[:3, :3] = R
    T[:3, 3]  = t
    return T
```

**Mathematical Derivation:**

The rotation matrix for a rotation around the y-axis by angle $\theta = -135° = -\frac{3\pi}{4}$ radians:

$$R_y(\theta) = \begin{bmatrix}
\cos(\theta) & 0 & -\sin(\theta) \\
0 & 1 & 0 \\
\sin(\theta) & 0 & \cos(\theta)
\end{bmatrix}$$

The translation vector must satisfy: $R \cdot c_{world} + t = 0_{camera}$

Therefore: $t = -R \cdot c_{world}$

This ensures that when we transform the camera origin from world coordinates, we get the camera's own origin $(0, 0, 0)$.

**Why -135°?**
- The camera is positioned 45° off the world z-axis in the xz-plane
- The camera looks back toward the origin (180° from its position vector)
- Combined: 45° + 90° = 135° rotation from world z-axis to camera z-axis
- We use -135° (negative) because we're transforming FROM world TO camera (inverse rotation)
- In radians: −135° = -$\frac{3\pi}{4}$ radians

---

## Part 2: Filtering

### Question 2.1: Convolution Implementation

The convolution operation is defined mathematically as:

$$(f*h)[m,n]=\sum_{i=-\infty}^\infty\sum_{j=-\infty}^\infty f[i,j]\cdot h[m-i,n-j]$$

**Solution:**

```python
def conv_nested(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    # Flip the kernel for convolution (not correlation)
    kernel_flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    
    # Calculate padding needed
    pad_h = Hk // 2
    pad_w = Wk // 2
    
    # Pad the image to handle boundaries
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Perform convolution using nested loops
    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    out[i, j] += kernel_flipped[m, n] * padded_image[i + m, j + n]
    
    return out
```

#### Why Do We Need to Flip the Kernel?

**Mathematical Reason:**

Convolution is defined as:
$$(f*h)[m,n]=\sum_{i}\sum_{j} f[i,j]\cdot h[m-i,n-j]$$

Notice the **negative signs**: $h[m-i, n-j]$. This means we're accessing the kernel in reverse order.

**Visual Example:**

Without flipping (this is called **correlation**):
```
Image patch:        Kernel:           Result:
[1  2  3]          [a  b  c]         
[4  5  6]    ×     [d  e  f]    =    1×a + 2×b + 3×c + 4×d + 5×e + 6×f + ...
[7  8  9]          [g  h  i]
```

With flipping (true **convolution**):
```
Image patch:        Kernel (flipped):  Result:
[1  2  3]          [i  h  g]         
[4  5  6]    ×     [f  e  d]    =    1×i + 2×h + 3×g + 4×f + 5×e + 6×d + ...
[7  8  9]          [c  b  a]
```

**Why Does This Matter?**

1. **Mathematical correctness**: Convolution is commutative ($f*h = h*f$), but correlation is not
2. **Filter design**: Edge detection filters like Sobel are designed assuming convolution
3. **Example with edge detection**:

```
Sobel X filter (detects vertical edges):
[-1  0  +1]
[-2  0  +2]
[-1  0  +1]

Without flipping: detects edges going left-to-right (dark to bright)
With flipping: detects edges going right-to-left (as intended)
```

**What It Improves:**
- Ensures filters work as mathematically intended
- Makes convolution commutative and associative
- Critical for separable filters and certain mathematical properties

#### Is Padding Used to Avoid Edge/Corner Problems?

**Yes, exactly!** Padding solves the boundary problem.

**The Problem Without Padding:**

```
Original 5×5 image with 3×3 kernel:

? ? ? ? ?        Can't compute output for corners
? O O O ?        because kernel extends outside image
? O O O ?        
? O O O ?        O = can compute output
? ? ? ? ?        ? = can't compute output

Result: 3×3 output (lost 2 rows and 2 columns)
```

**With Padding:**

```
Padded image (added 1 pixel on each side):

0 0 0 0 0 0 0
0 1 2 3 4 5 0
0 6 7 8 9 A 0     Now we can compute convolution
0 B C D E F 0     for every pixel in the original image
0 G H I J K 0
0 L M N O P 0     Result: 5×5 output (same as input)
0 0 0 0 0 0 0
```

**Padding Amount:**
```python
pad_h = Hk // 2  # For 3×3 kernel: pad by 1
pad_w = Wk // 2  # For 5×5 kernel: pad by 2
```

This ensures: **Output size = Input size**

---

### Question 2.2: Separable Convolution Theory

**Question (i): Direct 2D Convolution Complexity**

For an $M_1 \times N_1$ image and an $M_2 \times N_2$ filter:

**Number of multiplications** = $M_1 \times N_1 \times M_2 \times N_2$

**Question (ii): Separable Convolution Complexity**

For separable filters where $F = F_1 \cdot F_2$:

**Number of multiplications** = $M_1 \times N_1 \times M_2 + M_1 \times N_1 \times N_2$

This can be simplified to: $M_1 \times N_1 \times (M_2 + N_2)$

**Computational Advantage:**

The speedup factor is approximately:
$$\frac{M_2 \times N_2}{M_2 + N_2}$$

For a $5 \times 5$ kernel, this gives a speedup of $\frac{25}{10} = 2.5\times$

#### Why Is the Number of Multiplications Every Variable Multiplied?

**2D Convolution Complexity:**

Let's break down what happens:

```
Image: M₁ × N₁ (e.g., 100 × 100)
Kernel: M₂ × N₂ (e.g., 5 × 5)

For EACH output pixel:
├─ We need to compute: sum of (image_patch × kernel)
└─ This requires: M₂ × N₂ multiplications

Total output pixels: M₁ × N₁

Total multiplications = M₁ × N₁ × M₂ × N₂
```

**Concrete Example:**

```
100×100 image with 5×5 kernel:

- Output has 100 × 100 = 10,000 pixels
- Each output pixel needs 5 × 5 = 25 multiplications
- Total: 10,000 × 25 = 250,000 multiplications
```

**Why All Variables Multiply:**
- **M₁ × N₁**: Number of output positions (every pixel)
- **M₂ × N₂**: Work per output position (kernel size)
- Combined: **M₁ × N₁ × M₂ × N₂**

#### Difference Between 2D and 1D Convolution

**2D Convolution (Standard):**

```python
# Apply 2D kernel all at once
kernel_2d = [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]

result = convolve2d(image, kernel_2d)
# Cost: M₁ × N₁ × M₂ × N₂ = 100 × 100 × 3 × 3 = 90,000 multiplications
```

**1D Convolution (Separable):**

```python
# Apply two 1D kernels sequentially
kernel_vertical   = [[1], [2], [1]]  # 3×1
kernel_horizontal = [[1, 2, 1]]       # 1×3

temp = convolve1d(image, kernel_vertical, axis=0)    # vertical pass
result = convolve1d(temp, kernel_horizontal, axis=1)  # horizontal pass

# Cost: M₁ × N₁ × M₂ + M₁ × N₁ × N₂ 
#     = 100 × 100 × 3 + 100 × 100 × 3 = 60,000 multiplications
```

**Visual Representation:**

```
2D Convolution:
[Image] × [2D Kernel] → [Output]
         (one step)

1D Separable Convolution:
[Image] × [Vertical Kernel] → [Temp] × [Horizontal Kernel] → [Output]
         (step 1)                      (step 2)
```

**Key Differences:**

| Aspect | 2D Convolution | 1D Separable |
|--------|----------------|--------------|
| Kernel Type | Single 2D matrix | Two 1D vectors |
| Steps | One pass | Two passes |
| Cost | M₁×N₁×M₂×N₂ | M₁×N₁×(M₂+N₂) |
| When Possible | Always | Only if kernel is separable |

#### Does 2D Convolution Have Higher Quality Output?

**No! The output quality is identical IF the kernel is separable.**

**Proof:**
```python
# These produce EXACTLY the same result:
k_2d = [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]

k1 = [[1], [2], [1]]
k2 = [[1, 2, 1]]

assert np.all(k1 @ k2 == k_2d)  # True!

# Therefore:
output_2d = convolve2d(image, k_2d)
output_1d = convolve1d(convolve1d(image, k1), k2)

assert np.allclose(output_2d, output_1d)  # True!
```

**Why Use Separable Convolution?**
- **Speed**: 2.5× faster for 5×5 kernels
- **Quality**: Identical output
- **Limitation**: Only works if kernel can be separated

**When Can't We Separate?**

Some kernels are NOT separable:
```python
# This CANNOT be written as k1 × k2:
non_separable = [[1,  0, -1],
                 [0,  0,  0],
                 [-1, 0,  1]]
```

---

### Question 2.3: Separable Convolution Application

For the Gaussian kernel:
$$\begin{bmatrix}
1 & 4 & 6 & 4 & 1\\
4 & 16 & 24 & 16 & 4\\
6 & 24 & 36 & 24 & 6\\
4 & 16 & 24 & 16 & 4\\
1 & 4 & 6 & 4 & 1
\end{bmatrix}$$

**Solution:**

```python
k1 = np.array([1, 4, 6, 4, 1]).reshape(5, 1)  # Column vector
k2 = np.array([1, 4, 6, 4, 1]).reshape(1, 5)  # Row vector
```

**Verification:**

$$k_1 \cdot k_2 = \begin{bmatrix}1\\4\\6\\4\\1\end{bmatrix} \cdot \begin{bmatrix}1 & 4 & 6 & 4 & 1\end{bmatrix} = \text{Gaussian Kernel}$$

This decomposition allows us to apply two 1D convolutions instead of one 2D convolution, significantly reducing computational cost.

---

## Part 3: SIFT Descriptor

### Question 3.1: Computing a SIFT Descriptor

The SIFT (Scale-Invariant Feature Transform) descriptor creates a unique fingerprint for image patches by analyzing gradient information.

#### What is SIFT?

**SIFT = Scale-Invariant Feature Transform**

**Purpose:** Create a unique "fingerprint" for parts of an image that is:
- **Scale-invariant**: Works at different zoom levels
- **Rotation-invariant**: Works regardless of rotation
- **Illumination-invariant**: Works in different lighting

**How It Works (Simplified):**

```
1. Find interesting points (keypoints)
   └─ Corners, edges, texture changes

2. For each keypoint, extract a patch (16×16 pixels)

3. Compute gradients (magnitude & orientation)

4. Build histogram of gradients
   ├─ Divide patch into 4×4 grid (16 cells)
   ├─ Each cell: 8-bin orientation histogram
   └─ Result: 128-dimensional descriptor

5. Use descriptor for matching
   └─ Similar objects have similar descriptors
```

**Real-World Analogy:**

Imagine describing a person's face:
- **Bad descriptor**: "Has eyes, nose, mouth" (too general)
- **SIFT-like descriptor**: "Eyes 5cm apart, nose angles 30° left, mouth 3cm wide, wrinkle patterns at 45° angles..." (very specific)

**Why Use SIFT?**

```python
# Without SIFT (raw pixels):
shoe1 = [255, 200, 180, ...]  # 784 numbers
shoe2 = [250, 195, 175, ...]  # Similar but not identical

# With SIFT (descriptor):
shoe1_sift = [0.02, 0.15, 0.08, ...]  # 128 numbers
shoe2_sift = [0.02, 0.15, 0.08, ...]  # Much more similar!
```

**Applications:**
- Image matching (finding same object in different photos)
- Panorama stitching
- Object recognition
- 3D reconstruction

### Function 1: `compute_grad_mag_ori`

Computes gradient magnitude and orientation for each pixel in the patch.

```python
def compute_grad_mag_ori(patch):
    # Compute gradients using numpy
    Gx, Gy = np.gradient(patch)
    
    # Gradient magnitude: M = sqrt(Gx² + Gy²)
    grad_mag = np.sqrt(Gx**2 + Gy**2)
    
    # Gradient orientation: θ = atan2(Gy, Gx)
    # Convert from radians to degrees
    grad_ori = (np.arctan2(Gy, Gx)) * (180/np.pi)
    
    return grad_mag, grad_ori
```

**Mathematical Formulation:**

Gradient magnitude: $M = \sqrt{G_x^2 + G_y^2}$

Gradient orientation: $\theta = \arctan2(G_y, G_x) \cdot \frac{180°}{\pi}$

#### What is Gradient Magnitude?

**Definition:** Gradient magnitude measures **how quickly** pixel intensity changes.

**Formula:**
$$M = \sqrt{G_x^2 + G_y^2}$$

where:
- $G_x$ = change in x-direction (horizontal)
- $G_y$ = change in y-direction (vertical)

**Visual Example:**

```
Image patch:
[10  10  10]
[10  50  90]    ← Notice the bright spot
[10  10  10]

Gradients at center pixel:
Gx = 40 (change left-to-right)
Gy = 0  (no change top-to-bottom)

Magnitude = √(40² + 0²) = 40
```

**What It Tells Us:**

- **High magnitude**: Strong edge or texture (rapid intensity change)
- **Low magnitude**: Smooth area (little change)
- **Zero magnitude**: Flat region (no change)

**Real-World Analogy:**
Think of an image as a landscape (bright = high elevation):
- **Magnitude = steepness of the hill**
- Flat plains = low magnitude
- Steep cliffs = high magnitude

#### What is Gradient Orientation?

**Definition:** Gradient orientation indicates the **direction** of the intensity change.

**Formula:**
$$\theta = \arctan2(G_y, G_x) \times \frac{180°}{\pi}$$

**Visual Example:**

```
Different edge orientations:

Vertical edge:        Horizontal edge:      Diagonal edge:
[10  90]             [10  10]              [10  50]
[10  90]             [90  90]              [50  90]

Gx = 80, Gy = 0      Gx = 0, Gy = 80       Gx = 40, Gy = 40
θ = 0°               θ = 90°               θ = 45°
(→ direction)        (↑ direction)         (↗ direction)
```

**What It Tells Us:**

- **0°**: Intensity increases toward the right (→)
- **90°**: Intensity increases upward (↑)
- **180°/-180°**: Intensity increases toward the left (←)
- **-90°/270°**: Intensity increases downward (↓)

**Real-World Analogy:**
- **Orientation = compass direction of the steepest slope**
- If you're on a hillside, which way would water flow?


### Function 2: `compute_histogram_of_gradients`

Creates an 8-bin histogram of gradient orientations, weighted by magnitude.

```python
def compute_histogram_of_gradients(grad_ori, weight, nbins):
    hist = np.zeros(nbins)
    bin_width = 360.0 / nbins  # 45 degrees per bin for nbins=8
    
    orientations = grad_ori.flatten()
    weights = weight.flatten()
    
    for ori, mag in zip(orientations, weights):
        # Determine which bin this orientation falls into
        bin_index = int(ori // bin_width)
        
        # Handle edge case where orientation is exactly 360
        if bin_index >= nbins:
            bin_index = nbins - 1
            
        # Add weighted contribution to histogram
        hist[bin_index] += mag
    
    return hist
```

#### Is `compute_histogram_of_gradients` Just Assigning Orientations to Bins?

**Yes, exactly! But with a crucial addition: weighting by magnitude.**

**What It Does:**

```python
def compute_histogram_of_gradients(grad_ori, weight, nbins):
    # 1. Create empty histogram
    hist = np.zeros(8)
    
    # 2. For each gradient:
    for orientation, magnitude in zip(grad_ori, weight):
        # 3. Find which bin (which 1/8 piece)
        bin_index = int(orientation // 45)
        
        # 4. Add magnitude to that bin (not just count +1)
        hist[bin_index] += magnitude
    
    return hist
```

**Example:**

```python
Gradients in a 4×4 window:
Orientation: [10°, 15°, 50°, 95°, 280°]
Magnitude:   [20,  15,  30,  25,  10]

Processing:
- 10° → Bin 0, add 20  →  hist[0] = 20
- 15° → Bin 0, add 15  →  hist[0] = 35
- 50° → Bin 1, add 30  →  hist[1] = 30
- 95° → Bin 2, add 25  →  hist[2] = 25
- 280° → Bin 6, add 10 →  hist[6] = 10

Final histogram:
[35, 30, 25, 0, 0, 0, 10, 0]
```

**Key Point:** We don't just count how many gradients fall in each bin; we **sum their magnitudes**. Strong edges contribute more!

**Binning Strategy:**
- 8 bins covering 360°
- Each bin covers 45° (bin 0: 0-44°, bin 1: 45-89°, etc.)
- Contributions are weighted by gradient magnitude

#### Why 8 Bins?

**Historical and Practical Reasons:**

1. **Balance between precision and robustness:**
   - Too few bins (e.g., 4): Not distinctive enough
   - Too many bins (e.g., 32): Too sensitive to small rotations
   - 8 bins: Sweet spot for most applications

2. **Coverage:**
   ```
   360° ÷ 8 bins = 45° per bin
   
   Bin 0: 0° - 44°    (→)
   Bin 1: 45° - 89°   (↗)
   Bin 2: 90° - 134°  (↑)
   Bin 3: 135° - 179° (↖)
   Bin 4: 180° - 224° (←)
   Bin 5: 225° - 269° (↙)
   Bin 6: 270° - 314° (↓)
   Bin 7: 315° - 359° (↘)
   ```

3. **Computational efficiency:**
   - 8 is a power of 2 (good for computers)
   - 4×4 grid × 8 bins = 128 dimensions (manageable size)

**Visual Representation:**

```
        90° (↑)
         |
   135° ↖|↗ 45°
         |
180° ←---+---→ 0°
         |
   225° ↙|↘ 315°
         |
       270° (↓)
```

#### What Are Bins?

**Definition:** Bins are **categories** or **buckets** that group similar values together.

**Analogy - Age Groups:**
```
Instead of tracking exact ages:
23, 25, 28, 31, 33, 35, 38...

We use age bins:
Bin 1 (20-29): Count = 3
Bin 2 (30-39): Count = 4
```

**In SIFT Context:**

```python
# Gradient orientations (exact values):
orientations = [5°, 12°, 47°, 52°, 91°, 95°, 182°, 275°]

# Grouped into 8 bins (45° each):
Bin 0 (0-44°):     [5°, 12°]           → Count: 2
Bin 1 (45-89°):    [47°, 52°]          → Count: 2
Bin 2 (90-134°):   [91°, 95°]          → Count: 2
Bin 3 (135-179°):  []                  → Count: 0
Bin 4 (180-224°):  [182°]              → Count: 1
Bin 5 (225-269°):  []                  → Count: 0
Bin 6 (270-314°):  [275°]              → Count: 1
Bin 7 (315-359°):  []                  → Count: 0
```

**Purpose:**
- **Simplification**: Reduce continuous data to discrete categories
- **Robustness**: Small variations (5° vs 8°) fall in same bin
- **Compression**: Store summary instead of all individual values

#### Is the Histogram Calculated Per Category?

**Answer: No, the histogram is calculated per 4×4 window, not per clothing category.**

Let me clarify the structure:

```
One 16×16 Image Patch (e.g., one shoe image):
├─ Divided into 16 cells (4×4 grid)
│  ├─ Cell (0,0): 4×4 pixels → 1 histogram (8 bins)
│  ├─ Cell (0,1): 4×4 pixels → 1 histogram (8 bins)
│  ├─ Cell (0,2): 4×4 pixels → 1 histogram (8 bins)
│  └─ ... (16 cells total)
└─ Result: 16 histograms × 8 bins = 128 values
```

**Visual Breakdown:**

```
16×16 Image:
┌────────────────┐
│ □ □ □ □        │  Each □ = 4×4 cell
│ □ □ □ □        │  Each cell gets its own
│ □ □ □ □        │  8-bin histogram
│ □ □ □ □        │
└────────────────┘

Cell (0,0) histogram:      Cell (1,2) histogram:
Bin 0: ████ (4)            Bin 0: ██ (2)
Bin 1: ██████ (6)          Bin 1: ████████ (8)
Bin 2: ██ (2)              Bin 2: █ (1)
...                        ...
```

**Not Like This:**
```
❌ WRONG: One histogram per clothing category
    T-shirt category: [histogram]
    Shoe category: [histogram]
    
✓ CORRECT: One histogram per 4×4 cell within each image
    Image 1, Cell 1: [histogram]
    Image 1, Cell 2: [histogram]
    ...
    Image 1, Cell 16: [histogram]
```

#### Why Do We Flatten the Array?

**Reason:** To make iteration easier and faster.

**Before Flattening:**

```python
# 2D arrays - need nested loops
grad_ori = [[10, 20, 30],
            [40, 50, 60]]  # shape: (2, 3)

for i in range(2):
    for j in range(3):
        value = grad_ori[i][j]  # Tedious!
```

**After Flattening:**

```python
# 1D array - single loop
grad_ori_flat = [10, 20, 30, 40, 50, 60]  # shape: (6,)

for value in grad_ori_flat:
    # Much simpler!
```

**In the Code:**

```python
orientations = grad_ori.flatten()  # (4, 4) → (16,)
weights = grad_mag.flatten()       # (4, 4) → (16,)

for ori, mag in zip(orientations, weights):
    # Process each pixel easily
    bin_index = int(ori // bin_width)
    hist[bin_index] += mag
```

**Benefits:**
- Simpler code (one loop instead of two)
- Easier to use with `zip()`
- No performance loss (just changes memory layout)


#### Function 3: `generate_descriptors_from_patches`

Generates the complete 128-dimensional SIFT descriptor.

```python
def generate_descriptors_from_patches(image_patch):
    window_width = 4  # Divide 16×16 patch into 4×4 windows
    desc = np.zeros((window_width*window_width*8,))  # 4×4×8 = 128 dimensions
    
    # Compute gradients for entire patch
    grad_mag, grad_ori = compute_grad_mag_ori(image_patch)
    
    desc_idx = 0
    
    # Process each 4×4 window
    for i in range(0, 16, window_width):
        for j in range(0, 16, window_width):
            # Extract the 4×4 window
            mag_window = grad_mag[i:i+window_width, j:j+window_width]
            ori_window = grad_ori[i:i+window_width, j:j+window_width]
            
            # Compute histogram for this window
            hist = compute_histogram_of_gradients(ori_window, mag_window, nbins=8)
            
            # Add to descriptor
            desc[desc_idx:desc_idx+8] = hist
            desc_idx += 8
    
    # Normalize descriptor
    desc /= (np.linalg.norm(desc) + 1e-60)
    
    # Threshold large values for illumination invariance
    desc[desc > 0.2] = 0.2
    
    # Normalize again
    desc /= (np.linalg.norm(desc) + 1e-60)
    
    return desc
```

**SIFT Descriptor Structure:**
- Input: 16×16 image patch
- Divided into: 4×4 grid of cells (16 cells total)
- Each cell: 8-bin orientation histogram
- Output: 128-dimensional feature vector (4×4×8)

**Normalization Steps:**
1. L2 normalization of entire descriptor
2. Threshold values > 0.2 to achieve illumination invariance
3. L2 normalization again

#### Is the `desc` Variable a NumPy Array? What's Its Shape?

**Yes, `desc` is a NumPy array.**

**Shape Evolution:**

```python
# Initialization
desc = np.zeros((window_width*window_width*8,))
desc = np.zeros((4*4*8,))
desc = np.zeros((128,))  # Shape: (128,)

# It's a 1D array with 128 elements
```

**Structure:**

```python
desc[0:8]    = histogram from cell (0,0)
desc[8:16]   = histogram from cell (0,1)
desc[16:24]  = histogram from cell (0,2)
desc[24:32]  = histogram from cell (0,3)
desc[32:40]  = histogram from cell (1,0)
...
desc[120:128] = histogram from cell (3,3)

# Total: 16 cells × 8 bins/cell = 128 values
```

**Final Dataset Shape:**

```python
descriptors_np = np.array(descriptors)
print(descriptors_np.shape)  # (50, 128)

# 50 images in dataset
# Each image → 128-dimensional descriptor
```

**Visualization:**

```
One Image → One Descriptor (1D array):
[0.02, 0.15, 0.08, ..., 0.11]  ← 128 numbers
 └─┬─┘  └─┬─┘  └─┬─┘      └─┬─┘
Cell 1  Cell 1  Cell 1   Cell 16
Bin 0   Bin 1   Bin 2    Bin 7

All Images → 2D array:
[[0.02, 0.15, ..., 0.11],  ← Image 1
 [0.05, 0.18, ..., 0.09],  ← Image 2
 ...
 [0.03, 0.12, ..., 0.14]]  ← Image 50
```