import pyvista as pv
import numpy as np
import os
import sys
import warnings
from typing import Optional, Union, Tuple

# Suppress VTK warnings
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

# Alternative: suppress specific warning patterns
warnings.filterwarnings("ignore", category=UserWarning, module="pyvista")
warnings.filterwarnings("ignore", message=".*bad X server connection.*")
warnings.filterwarnings("ignore", message=".*EGL.*")
warnings.filterwarnings("ignore", message=".*OpenGL.*")

# Set up environment for headless rendering
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_PANEL'] = 'false'
os.environ['DISPLAY'] = ''
pv.OFF_SCREEN = True

import numpy as np
from collections import deque
import matplotlib.pyplot as plt



def count_clusters_simple_jax(field, threshold=0.75, connectivity=26):
    """
    JAX flood-fill cluster counter with periodic boundaries.
    Clusters voxels with values < threshold.
    """
    
    # Convert to numpy for flood-fill
    field_np = np.array(field)
    
    # Binarize: values < threshold become 1 (foreground)
    binary_field = (field_np < threshold).astype(int)
    nx, ny, nz = binary_field.shape
    
    # Visited array
    visited = np.zeros_like(binary_field, dtype=bool)
    
    # Cluster information
    clusters = []
    cluster_id = 0
    labeled_field = np.zeros_like(binary_field)
    
    def get_neighbors_periodic(x, y, z):
        """Get periodic neighbors based on connectivity"""
        neighbors = []
        
        if connectivity == 6:
            deltas = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        elif connectivity == 18:
            deltas = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if abs(dx) + abs(dy) + abs(dz) <= 2 and (dx, dy, dz) != (0, 0, 0):
                            deltas.append((dx, dy, dz))
        else:  # 26-connectivity
            deltas = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if (dx, dy, dz) != (0, 0, 0):
                            deltas.append((dx, dy, dz))
        
        for dx, dy, dz in deltas:
            nx_coord = (x + dx) % nx
            ny_coord = (y + dy) % ny  
            nz_coord = (z + dz) % nz
            neighbors.append((nx_coord, ny_coord, nz_coord))
        
        return neighbors
    
    def flood_fill(start_x, start_y, start_z):
        """Flood fill to find connected component"""
        queue = deque([(start_x, start_y, start_z)])
        cluster_size = 0
        
        while queue:
            x, y, z = queue.popleft()
            
            if visited[x, y, z] or binary_field[x, y, z] == 0:
                continue
                
            visited[x, y, z] = True
            labeled_field[x, y, z] = cluster_id + 1
            cluster_size += 1
            
            # Add neighbors to queue
            for nx_coord, ny_coord, nz_coord in get_neighbors_periodic(x, y, z):
                if not visited[nx_coord, ny_coord, nz_coord] and binary_field[nx_coord, ny_coord, nz_coord]:
                    queue.append((nx_coord, ny_coord, nz_coord))
        
        return cluster_size
    
    # Find all clusters
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if binary_field[i, j, k] and not visited[i, j, k]:
                    cluster_size = flood_fill(i, j, k)
                    # if cluster_size > 2:  # Only keep clusters larger than 2 voxels
                    clusters.append(cluster_size)
                    cluster_id += 1
    
    # Convert results to JAX arrays
    return len(clusters), np.array(labeled_field), np.array(clusters)

def analyze_clusters_over_time(result, threshold=-0.8, connectivity=26, time_indices=None):
    """
    Analyze clusters over time for 4D data (nt, nx, ny, nz).
    
    Parameters:
    -----------
    result : array-like
        4D array with shape (nt, nx, ny, nz, channels) or (nt, nx, ny, nz)
    threshold : float
        Threshold for clustering (values < threshold)
    connectivity : int
        Connectivity type (6, 18, or 26)
    time_indices : list or None
        Which time steps to analyze (if None, use all)
        
    Returns:
    --------
    time_data : dict
        Dictionary with time series data
    """
    
    # Handle different input shapes
    if result.ndim == 5:
        # Shape (nt, nx, ny, nz, channels) - extract concentration
        concentration_data = result[..., 0]
    elif result.ndim == 4:
        # Shape (nt, nx, ny, nz) - already concentration
        concentration_data = result
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {result.shape}")
    
    nt = concentration_data.shape[0]
    
    if time_indices is None:
        time_indices = list(range(nt))
    
    print(f"Analyzing clusters over {len(time_indices)} time steps")
    print(f"Threshold: < {threshold}, Connectivity: {connectivity}")
    print(f"Data shape: {concentration_data.shape}")
    print("-" * 50)
    
    # Storage for results
    time_data = {
        'time_indices': [],
        'num_clusters': [],
        'largest_cluster': [],
        'total_cluster_volume': [],
        'mean_cluster_size': [],
        'cluster_size_std': [],
        'volume_fraction': []
    }
    
    # Analyze each time step
    for i, t in enumerate(time_indices):
        if i % max(1, len(time_indices) // 10) == 0:  # Progress update
            print(f"Processing time step {t} ({i+1}/{len(time_indices)})")
        
        # Extract concentration field at time t
        concentration = concentration_data[t]
        
        # Count clusters
        num_clusters, labeled_field, cluster_sizes = count_clusters_simple_jax(
            field=concentration,
            threshold=threshold,
            connectivity=connectivity
        )
        
        # Compute statistics
        largest_cluster = int(np.max(cluster_sizes)) if len(cluster_sizes) > 0 else 0
        total_volume = int(np.sum(cluster_sizes)) if len(cluster_sizes) > 0 else 0
        mean_size = float(np.mean(cluster_sizes)) if len(cluster_sizes) > 0 else 0
        size_std = float(np.std(cluster_sizes)) if len(cluster_sizes) > 0 else 0
        
        # Volume fraction below threshold
        total_voxels = concentration.size
        volume_fraction = total_volume / total_voxels
        
        # Store results
        time_data['time_indices'].append(t)
        time_data['num_clusters'].append(num_clusters)
        time_data['largest_cluster'].append(largest_cluster)
        time_data['total_cluster_volume'].append(total_volume)
        time_data['mean_cluster_size'].append(mean_size)
        time_data['cluster_size_std'].append(size_std)
        time_data['volume_fraction'].append(volume_fraction)
    
    print("Analysis complete!")
    return time_data

def plot_cluster_evolution(time_data, title_prefix="Cluster Evolution", figsize=(15, 10)):
    """
    Plot cluster evolution over time.
    
    Parameters:
    -----------
    time_data : dict
        Output from analyze_clusters_over_time()
    title_prefix : str
        Prefix for plot titles
    figsize : tuple
        Figure size
    """
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"{title_prefix} Over Time", fontsize=16, fontweight='bold')
    
    time_indices = time_data['time_indices']
    
    # Plot 1: Number of clusters
    axes[0, 0].plot(time_indices, time_data['num_clusters'], 'b-o', markersize=4)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].set_title('Cluster Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Largest cluster size
    axes[0, 1].plot(time_indices, time_data['largest_cluster'], 'r-s', markersize=4)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Largest Cluster Size')
    axes[0, 1].set_title('Largest Cluster Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Volume fraction
    axes[0, 2].plot(time_indices, time_data['volume_fraction'], 'g-^', markersize=4)
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Volume Fraction')
    axes[0, 2].set_title('Volume Fraction Below Threshold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Mean cluster size
    axes[1, 0].plot(time_indices, time_data['mean_cluster_size'], 'm-d', markersize=4)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Mean Cluster Size')
    axes[1, 0].set_title('Average Cluster Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Total cluster volume
    axes[1, 1].plot(time_indices, time_data['total_cluster_volume'], 'c-v', markersize=4)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Total Cluster Volume')
    axes[1, 1].set_title('Total Volume in Clusters')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Cluster size standard deviation
    axes[1, 2].plot(time_indices, time_data['cluster_size_std'], 'orange', marker='h', markersize=4)
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Cluster Size Std Dev')
    axes[1, 2].set_title('Cluster Size Variability')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_cluster_count_simple(time_data, figsize=(10, 6)):
    """
    Simple plot of just cluster count vs time.
    """
    
    plt.figure(figsize=figsize)
    plt.plot(time_data['time_indices'], time_data['num_clusters'], 
             'b-o', linewidth=2, markersize=6, markerfacecolor='lightblue', 
             markeredgecolor='blue', markeredgewidth=1.5)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.title('Cluster Count Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_cluster_summary(time_data):
    """
    Print summary statistics of cluster evolution.
    """
    
    print("\n" + "="*60)
    print("CLUSTER EVOLUTION SUMMARY")
    print("="*60)
    
    time_indices = time_data['time_indices']
    num_clusters = time_data['num_clusters']
    largest_clusters = time_data['largest_cluster']
    volume_fractions = time_data['volume_fraction']
    
    print(f"Time range: {min(time_indices)} to {max(time_indices)} ({len(time_indices)} steps)")
    print(f"Cluster count - Min: {min(num_clusters)}, Max: {max(num_clusters)}, Final: {num_clusters[-1]}")
    print(f"Largest cluster - Min: {min(largest_clusters)}, Max: {max(largest_clusters)}, Final: {largest_clusters[-1]}")
    print(f"Volume fraction - Min: {min(volume_fractions):.4f}, Max: {max(volume_fractions):.4f}, Final: {volume_fractions[-1]:.4f}")
    
    # Find key events
    max_clusters_idx = np.argmax(num_clusters)
    max_largest_idx = np.argmax(largest_clusters)
    
    print(f"\nKey events:")
    print(f"  Maximum clusters ({max(num_clusters)}) at time step {time_indices[max_clusters_idx]}")
    print(f"  Maximum largest cluster ({max(largest_clusters)}) at time step {time_indices[max_largest_idx]}")

# Example usage function
def analyze_your_simulation(result, threshold=-0.8, time_step_interval=1, plot=False):
    """
    Analyze your specific simulation result.
    
    Parameters:
    -----------
    result : array
        Your simulation result (shape: nt, nx, ny, nz, channels)
    threshold : float
        Threshold for clustering
    time_step_interval : int
        Analyze every N time steps (for faster processing)
    """
    
    # Select time steps to analyze
    nt = result.shape[0]
    time_indices = list(range(0, nt, time_step_interval))
    
    print(f"Analyzing simulation with shape: {result.shape}")
    print(f"Selected {len(time_indices)} time steps for analysis")
    
    # Run analysis
    time_data = analyze_clusters_over_time(
        result=result, 
        threshold=threshold, 
        connectivity=26, 
        time_indices=time_indices
    )
    
    if plot == True:
        # Print summary
        print_cluster_summary(time_data)

        # Create plots
        plot_cluster_evolution(time_data, title_prefix=f"Cahn-Hilliard (threshold < {threshold})")
        plot_cluster_count_simple(time_data)
    
    return time_data












































def numpy_to_pyvista_grid(data_3d: np.ndarray, 
                        visualization_type: str = 'slice',  # 'volume', 'slice', 'surface'
                         spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                         origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         add_padding: bool = True,
                         verbose: bool = False) -> pv.StructuredGrid:
    """
    Convert 3D numpy array to PyVista StructuredGrid
    
    Args:
        data_3d: 3D numpy array
        spacing: Grid spacing in x, y, z directions
        origin: Grid origin point
        add_padding: Add 2-voxel padding around data (like original VTK function)
        verbose: Print debug information
    
    Returns:
        PyVista StructuredGrid object
    """
    
    # Add padding if requested (matches original VTK creation)
    if add_padding:
        cover_len = data_3d.shape[-1] + 2
        padded_data = np.zeros((cover_len, cover_len, cover_len))
        padded_data[1:-1, 1:-1, 1:-1] = data_3d
        data_3d = padded_data
    
    # Create structured grid
    nx, ny, nz = data_3d.shape
    
    # Create coordinate arrays
    x = np.arange(nx) * spacing[0] + origin[0]
    y = np.arange(ny) * spacing[1] + origin[1]
    z = np.arange(nz) * spacing[2] + origin[2]
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create PyVista structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    
    # Add the data as point data (flatten in Fortran order to match VTK)
    if nx==16 and visualization_type=='slice':
        grid.cell_data['ScalarField'] = data_3d[:-1, :-1, :-1].flatten(order='F')
    else:
        grid.point_data['ScalarField'] = data_3d.flatten(order='F')
        
    return grid

def create_transparent_reference_grid(target_size: int = 64,
                                    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> pv.StructuredGrid:
    """
    Create a transparent reference grid to maintain consistent plot dimensions
    
    Args:
        target_size: Size of the reference grid (cubic)
        spacing: Grid spacing in x, y, z directions
        origin: Grid origin point
    
    Returns:
        PyVista StructuredGrid object with transparent data
    """
    # Create coordinate arrays for the reference grid
    x = np.arange(target_size) * spacing[0] + origin[0]
    y = np.arange(target_size) * spacing[1] + origin[1]
    z = np.arange(target_size) * spacing[2] + origin[2]
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create PyVista structured grid
    ref_grid = pv.StructuredGrid(X, Y, Z)
    
    # Add transparent/neutral scalar field (zeros or very small values)
    transparent_data = np.zeros((target_size, target_size, target_size))
    ref_grid.point_data['TransparentField'] = transparent_data.flatten(order='F')
    
    return ref_grid

def visualize_surface_field(data_3d: np.ndarray,
                           visualization_type: str = 'surface',  # 'volume', 'slice', 'surface'
                           opacity_mapping: Optional[list] = None,
                           color_mapping: Optional[list] = None,
                           output_image: Optional[str] = None,
                           title: str = None,
                           colormap: str = 'bwr',
                           background_color: Tuple[float, float, float] = (1., 1., 1.),
                           window_size: Tuple[int, int] = (1200, 1200),
                           auto_camera: bool = True,
                           show_scalar_bar: bool = True,
                           clim: Optional[Tuple[float, float]] = None,
                           verbose: bool = False,
                           zoom_size: float = 1.,
                           surface_level: float = -0.8,
                           use_reference_grid: bool = True,
                           reference_grid_size: int = 64) -> bool:
    """
    Create surface field visualization (volume rendering or surface plots)
    
    Args:
        data_3d: 3D numpy array
        visualization_type: Type of visualization ('volume', 'slice', 'surface')
        opacity_mapping: List of [value, opacity] pairs for volume rendering
        color_mapping: List of [value, r, g, b] for custom color mapping
        output_image: Output image path (optional)
        title: Visualization title
        colormap: Color map for visualization
        background_color: Background color as RGB tuple (0-1 range)
        window_size: Window size as (width, height)
        auto_camera: Use automatic camera positioning
        show_scalar_bar: Show color bar
        clim: Color limits as (min, max)
        verbose: Print debug information
        zoom_size: Zoom factor for camera
        surface_level: Isosurface level for surface visualization
        use_reference_grid: Add transparent reference grid for consistent sizing
        reference_grid_size: Size of the reference grid
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        if verbose:
            print(f"📊 Processing numpy data of shape: {data_3d.shape}")
            print(f"  Data range: {data_3d.min():.3f} to {data_3d.max():.3f}")
        
        # Convert to PyVista grid
        if visualization_type=='surface':
            grid = numpy_to_pyvista_grid(data_3d, visualization_type, verbose=verbose)
        else:
            grid = numpy_to_pyvista_grid(data_3d, visualization_type, add_padding=False, verbose=verbose)
        
        if verbose:
            print(f"  Grid created: {grid.n_points:,} points, {grid.n_cells:,} cells")
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.background_color = background_color
        
        # Create and add transparent reference grid if requested
        if use_reference_grid:
            ref_grid = create_transparent_reference_grid(
                target_size=reference_grid_size,
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0)
            )
            
            # Add transparent reference grid (invisible but affects bounds)
            plotter.add_mesh(
                ref_grid,
                scalars='TransparentField',
                opacity=0.0,  # Completely transparent
                show_scalar_bar=False,
                pickable=False  # Don't interfere with interactions
            )
            
            if verbose:
                print(f"  Reference grid added: {ref_grid.n_points:,} points (transparent)")
        
        # Set color limits if not provided
        if clim is None:
            clim = (-0.8, 0.8)
        
        # Determine bounds for camera and bounding box
        if use_reference_grid:
            # Use reference grid bounds for consistent sizing
            data_bounds = (0, reference_grid_size-1, 0, reference_grid_size-1, 0, reference_grid_size-1)
            data_center = np.array([reference_grid_size/2, reference_grid_size/2, reference_grid_size/2])
            data_size = reference_grid_size
        else:
            # Use actual data bounds
            if visualization_type == 'slice':
                data_center = (np.array(data_3d.shape)-1)/2
                data_bounds = (0,data_3d.shape[0]-1, 0, data_3d.shape[0]-1, 0, data_3d.shape[0]-1)
                data_size = data_3d.shape[0]-1
            else:
                data_center = (np.array(data_3d.shape))/2
                data_bounds = (1,data_3d.shape[0]+0.2, 1, data_3d.shape[0]+0.2, 1, data_3d.shape[0]+0.2)
                data_size = data_3d.shape[0]
        
        if visualization_type == 'volume':
            # Volume rendering
            if verbose:
                print("  Creating volume rendering...")
            
            # Default opacity mapping if not provided
            if opacity_mapping is None:
                data_range = clim[1] - clim[0]
                opacity_mapping = [
                    [clim[0], 0.0],                    # Minimum: transparent
                    [clim[0] + 0.2 * data_range, 0.1], # Low values: slightly visible
                    [clim[0] + 0.5 * data_range, 0.3], # Mid values: semi-transparent
                    [clim[0] + 0.8 * data_range, 0.7], # High values: more opaque
                    [clim[1], 1.0]                     # Maximum: fully opaque
                ]
            
            # Add volume
            plotter.add_volume(
                grid,
                scalars='ScalarField',
                cmap=colormap,
                opacity=opacity_mapping,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={
                    'title': 'Field Value',
                    'label_font_size': 30,
                    'title_font_size': 14,
                    'fmt': '%.1f',
                    'width': 0.3,
                    'height': 0.1,
                    'position_x': 0.6,
                    'position_y': 0.05,
                    'color': 'white' if background_color[0] < 0.5 else 'black'
                }
            )
            
        elif visualization_type == 'slice':
            # Slice planes through the volume
            if verbose:
                print("  Creating slice planes...")
            
            # Create slices at front faces
            bounds = grid.bounds
            print("  Creating front slices...")
            print(f"  Grid bounds: {bounds}")
            slices = grid.slice(normal='x', origin=[bounds[1], 0, 0])  # Front X face
            slices += grid.slice(normal='y', origin=[0, bounds[3], 0])  # Front Y face
            slices += grid.slice(normal='z', origin=[0, 0, bounds[5]])  # Front Z face
            
            plotter.add_mesh(
                slices,
                cmap=colormap,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={
                    'title': None,
                    'label_font_size': 30,
                    'title_font_size': 14,
                    'fmt': '%.1f',
                    'width': 0.3,
                    'height': 0.1,
                    'position_x': 0.62,
                    'position_y': 0.05,
                    'color': 'white' if background_color[0] < 0.5 else 'black'
                },
                smooth_shading=False,
                lighting=False,
            )
            
        elif visualization_type == 'surface':
            # Surface rendering with multiple isosurfaces
            if verbose:
                print("  Creating surface rendering...")
            
            # Create multiple isosurfaces
            iso_values = np.asarray([surface_level])
            colors = ['blue', (227, 245, 127), (245, 127, 216), 'yellow', 'red']
            
            if verbose:
                print(f"  Using isosurface values: {iso_values}")
            
            for i, iso_val in enumerate(iso_values):
                contour = grid.contour(isosurfaces=[iso_val], scalars='ScalarField')
                if contour.n_points > 0:
                    opacity = 1 - 0.35 * i 
                    plotter.add_mesh(
                        contour,
                        color=colors[i],
                        opacity=opacity,
                        smooth_shading=True,
                        show_scalar_bar=(show_scalar_bar and i == 0),
                        scalar_bar_args={
                            'title': 'Field Value',
                            'label_font_size': 30,
                            'title_font_size': 14,
                            'fmt': '%.1f',
                            'width': 0.3,
                            'height': 0.1,
                            'position_x': 0.6,
                            'position_y': 0.05,
                            'color': 'white' if background_color[0] < 0.5 else 'black'
                        }
                    )
        
        # Camera positioning
        if auto_camera:
            plotter.reset_camera()
            
            camera_distance = data_size * 1.8
            
            # Set isometric view
            camera_pos = [
                data_center[0] + camera_distance * 0.577,
                data_center[1] + camera_distance * 0.577,
                data_center[2] + camera_distance * 0.577 
            ]
            
            # Shift the focal point for better view
            vertical_shift = data_size * 0.1
            shifted_focal_point = [
                data_center[0],
                data_center[1], 
                data_center[2] - vertical_shift
            ]
            
            plotter.camera_position = camera_pos
            plotter.camera.focal_point = shifted_focal_point
            plotter.camera.up = [0, 0, 1]
            plotter.camera.zoom(zoom_size)
            
            if verbose:
                print(f"  Camera position: {camera_pos}")
                print(f"  Focal point: {shifted_focal_point}")
        
        # Add bounding box
        outline = pv.Box(bounds=data_bounds)
        plotter.add_mesh(
            outline,
            style='wireframe',
            color='white' if background_color[0] < 0.5 else 'black',
            line_width=2,
            opacity=1
        )
        
        # Enhanced lighting for better visualization
        if visualization_type == 'surface':
            plotter.add_light(pv.Light(position=(100, 100, 200), intensity=0.7))
            plotter.add_light(pv.Light(position=(-100, -100, 100), intensity=0.2))
        
        # Save screenshot if requested
        if output_image:
            plotter.screenshot(output_image, transparent_background=False)
            if verbose:
                print(f"✓ Screenshot saved to: {output_image}")
        
        plotter.close()
        
        if verbose:
            print(f"✓ Surface field visualization completed successfully")
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

def batch_visualize_surface_fields(file_configs: list,
                                  base_path: str = ".",
                                  verbose: bool = False) -> dict:
    """
    Batch process multiple NPY files with surface field visualization
    
    Args:
        file_configs: List of dictionaries with file configurations
        base_path: Base path for NPY files
        verbose: Print debug information
    
    Returns:
        Dictionary with results
    """
    
    results = {'successful': 0, 'failed': 0, 'files': []}
    
    for config in file_configs:
        file_path = os.path.join(base_path, config['file'])
        output_file = os.path.join(base_path, config['output_file'])
        
        if verbose:
            print(f"\n🔄 Processing: {config['file']}")
        
        success = load_and_visualize_surface_field(
            npy_file=file_path,
            visualization_type=config.get('visualization_type', 'surface'),
            data_index=config.get('data_index', (0, -1, ...)),
            output_image=output_file,
            title=config.get('title', None),
            clim=config.get('clim', (-0.8, 0.8)),
            verbose=verbose
        )
        
        results['files'].append({
            'file': config['file'],
            'success': success,
            'output': output_file if success else None
        })
        
        if success:
            results['successful'] += 1
        else:
            results['failed'] += 1
    
    if verbose:
        print(f"\n📊 Batch processing complete:")
        print(f"  ✓ Successful: {results['successful']}")
        print(f"  ✗ Failed: {results['failed']}")
    
    return results

def visualize_your_surface_data(verbose: bool = False):
    """
    Example function showing how to visualize your datasets with surface fields
    
    Args:
        verbose: Print debug information
    """
    
    # Configuration for your datasets with different visualization types
    configs = [
        # {
        #     'file': '/work/08171/nvhai/ls6/LLNL_2025_intern/Fei_ver2_example/che3d_train_all_C_m0p5_t0_0p5_all_stage_fluxconvnext_ntyp1_hid64_nmp8_lr5e-4_b64_ntr9999999_noi0.02_f-999_EM0_wtE0_wtN0_wn11_fd6_sk1_D0_wl50_n_out2_nn_normalize_factor1_mode_grad_F1_multi_input_c0_pred_with_noise0/pd_T4s_64x64x64_long_nonoise.npy',
        #     'visualization_type': 'surface',
        #     'output_file': 'Figs/flux6_conservative_long.png',
        # },
        {
            # 'file' : '/work/08171/nvhai/ls6/LLNL_2025_intern/Fei_ver2_example/my_data_train_noise_1ic_all_C_m0p5_t0_0p5_all_stages_ver2/noise0.02/test/valid3D_noise0.02.npy',
            # 'data_index': (0, -1, ...,0),  # Adjust as needed
            'file': '/work/08171/nvhai/ls6/LLNL_2025_intern/Fei_ver2_example/che3d_aa_paraview/vtk_data/data/data_final.npy',
            'visualization_type': 'surface',  # Changed to surface
            'output_file': 'Figs/data_final.png',
        },
        # {
        #     'file': '/work/08171/nvhai/ls6/LLNL_2025_intern/Fei_ver2_example/my_data_train_noise_1ic_single_all_stages_ver2_C0p5/noise0.02/test_64x64x64/test3d_64x64x64.npy',
        #     'visualization_type': 'surface',  # Changed to volume
        #     'output_file': 'Figs/initial_data_final.png',
        # },
        {
            'file': '/work/08171/nvhai/ls6/LLNL_2025_intern/Fei_ver2_example/che3d_train_all_C_m0p5_t0_0p5_all_stage_fluxconvnext_ntyp1_hid64_nmp8_lr5e-4_b64_ntr9999999_noi0.02_f-999_EM0_wtE0_wtN0_wn11_fd31_sk1_D0_wl50_n_out2_nn_normalize_factor1_mode_grad_F1_multi_input_c0_pred_with_noise1/pd_T4s_64x64x64_long_nonoise.npy',
            'visualization_type': 'surface',   # Changed to surface
            'output_file': 'Figs/flux31_conservative_long.png',          
        },
        # {
        #     'file': '/work/08171/nvhai/ls6/LLNL_2025_intern/Fei_ver2_example/che3d_train_all_C_m0p5_t0_0p5_all_stage_fluxconvnext_ntyp1_hid64_nmp8_lr5e-4_b64_ntr9999999_noi0.02_f-999_EM0_wtE0_wtN0_wn20_fd31_sk1_D0_wl50_n_out2_nn_normalize_factor1_mode_grad_F1_multi_input_c0_pred_with_noise0/pd_T4s_64x64x64_long_nonoise.npy',
        #     'visualization_type': 'surface',
        #     'output_file': 'Figs/flux31_w20_conservative_long.png',          
        # }
    ]
    
    return batch_visualize_surface_fields(configs, verbose=verbose)

# Additional utility functions for advanced visualizations
def create_custom_opacity_map(data_range: Tuple[float, float], 
                             opacity_points: list = None) -> list:
    """
    Create custom opacity mapping for volume rendering
    
    Args:
        data_range: (min, max) of data values
        opacity_points: List of relative positions (0-1) and opacity values
    
    Returns:
        List of [value, opacity] pairs
    """
    if opacity_points is None:
        opacity_points = [(0.0, 0.0), (0.2, 0.1), (0.5, 0.3), (0.8, 0.7), (1.0, 1.0)]
    
    data_min, data_max = data_range
    data_span = data_max - data_min
    
    opacity_map = []
    for pos, opacity in opacity_points:
        value = data_min + pos * data_span
        opacity_map.append([value, opacity])
    
    return opacity_map


if __name__ == "__main__":
    # Only print when running directly
    print("Surface field visualization ready!")
    print("\nVisualization types available:")
    print("  'volume' - Volume rendering (like ParaView volume render)")
    print("  'slice' - Orthogonal slice planes")
    print("  'surface' - Multiple isosurface layers")
    print("\nUsage examples:")
    print("  load_and_visualize_surface_field('data.npy', visualization_type='slice', verbose=True)")
    print("  visualize_your_surface_data(verbose=True)  # Process all your datasets")
    
    # Run with verbose output when executed directly
    visualize_your_surface_data(verbose=True)











































#! # This code is part of a module for visualizing 3D box surfaces and animations.

def create_3d_box_surfaces(volume):
    """Create the 6 surfaces of a 3D box with their coordinates and colors"""
    surfaces = []
    nx, ny, nz = volume.shape
    
    # Surface 1: x=0 (YZ plane)
    y, z = np.meshgrid(range(ny), range(nz), indexing='ij')
    x = np.zeros_like(y)
    colors = volume[0, :, :]
    surfaces.append((x, y, z, colors, 'x=0'))
    
    # Surface 2: x=max (YZ plane)
    y, z = np.meshgrid(range(ny), range(nz), indexing='ij')
    x = np.full_like(y, nx-1)
    colors = volume[-1, :, :]
    surfaces.append((x, y, z, colors, f'x={nx-1}'))
    
    # Surface 3: y=0 (XZ plane)
    x, z = np.meshgrid(range(nx), range(nz), indexing='ij')
    y = np.zeros_like(x)
    colors = volume[:, 0, :]
    surfaces.append((x, y, z, colors, 'y=0'))
    
    # Surface 4: y=max (XZ plane)
    x, z = np.meshgrid(range(nx), range(nz), indexing='ij')
    y = np.full_like(x, ny-1)
    colors = volume[:, -1, :]
    surfaces.append((x, y, z, colors, f'y={ny-1}'))
    
    # Surface 5: z=0 (XY plane)
    x, y = np.meshgrid(range(nx), range(ny), indexing='ij')
    z = np.zeros_like(x)
    colors = volume[:, :, 0]
    surfaces.append((x, y, z, colors, 'z=0'))
    
    # Surface 6: z=max (XY plane)
    x, y = np.meshgrid(range(nx), range(ny), indexing='ij')
    z = np.full_like(x, nz-1)
    colors = volume[:, :, -1]
    surfaces.append((x, y, z, colors, f'z={nz-1}'))
    
    return surfaces

def create_3d_box_animation(data, interval=100):
    """
    Create an animation of the 3D box surfaces evolution
    
    Parameters:
    - data: Array of shape [nt, nx, ny, nz]
    - interval: Time between frames in milliseconds
    
    Returns:
    - Animation object
    """
    nt, nx, ny, nz = data.shape
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up color normalization
    vmin, vmax = -0.9, 0.9
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.bwr
    
    # Initialize with first time step
    surfaces = create_3d_box_surfaces(data[0])
    surface_plots = []
    
    for x, y, z, colors, label in surfaces:
        surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(colors)), 
                              alpha=1.0, shade=False)
        surface_plots.append(surf)
    
    # Set labels and initial title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')
    title = ax.set_title(f'3D Box Surfaces - Timestep: 0')
    
    # Set fixed limits
    ax.set_xlim(0, nx-1)
    ax.set_ylim(0, ny-1)
    ax.set_zlim(0, nz-1)
    
    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Data Value')
    
    def update(frame):
        """Update function for animation"""
        # Clear previous surfaces
        ax.clear()
        
        # Create new surfaces for current time step
        surfaces = create_3d_box_surfaces(data[frame])
        
        for x, y, z, colors, label in surfaces:
            ax.plot_surface(x, y, z, facecolors=cmap(norm(colors)), 
                           alpha=1.0, shade=False)
        
        # Reset labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title.set_text(f'3D Box Surfaces - Timestep: {frame}')
        ax.set_xlim(0, nx-1)
        ax.set_ylim(0, ny-1)
        ax.set_zlim(0, nz-1)
        
        return [title]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=nt, 
                        interval=interval, blit=False)
    plt.close()  # Close the static figure
    return anim

def create_side_by_side_animation(data1, data2, labels=None, main_title=None, interval=100):
    """
    Create a side-by-side animation comparing two 3D box datasets
    
    Parameters:
    - data1: First dataset of shape [nt, nx, ny, nz]
    - data2: Second dataset of shape [nt, nx, ny, nz]
    - labels: Tuple of labels for the two datasets (default: ('Data1', 'Data2'))
    - main_title: Main title for the entire figure (appears above timestep)
    - interval: Time between frames in milliseconds
    
    Returns:
    - Animation object
    """
    if labels is None:
        labels = ('Data1', 'Data2')
    
    if main_title is None:
        main_title = "3D Box Surface Comparison"
    
    # Ensure both datasets have the same shape
    assert data1.shape == data2.shape, "Both datasets must have the same shape"
    
    nt, nx, ny, nz = data1.shape
    
    # Set up the figure with two 3D subplots
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Set up color normalization (use global min/max for consistency)
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    # Override with fixed range if needed
    vmin, vmax = -0.9, 0.9
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.bwr
    
    # Initialize both axes
    def setup_axis(ax, data, title, timestep=0):
        surfaces = create_3d_box_surfaces(data[timestep])
        
        for x, y, z, colors, label in surfaces:
            ax.plot_surface(x, y, z, facecolors=cmap(norm(colors)), 
                           alpha=1.0, shade=False)
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Remove tick labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set limits
        ax.set_xlim(0, nx-1)
        ax.set_ylim(0, ny-1)
        ax.set_zlim(0, nz-1)
        
        # Optional: Remove axis lines/panes (uncomment if desired)
        # ax.grid(False)
        # ax.set_axis_off()  # This removes everything including the box
        
        return ax.set_title(f'{title}')
    
    # Setup initial state
    title1 = setup_axis(ax1, data1, labels[0])
    title2 = setup_axis(ax2, data2, labels[1])
    
    # Add main title and timestep title
    fig.suptitle(f'{main_title}', fontsize=18, y=0.98)
    timestep_title = fig.text(0.5, 0.92, f'Timestep: 0 / {nt-1}', 
                             fontsize=14, ha='center', transform=fig.transFigure)
    
    # Add shared colorbar with manual positioning
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    
    # Method 1: Manual positioning with fig.add_axes([left, bottom, width, height])
    # Values are in figure coordinates (0 to 1)
    cbar_ax = fig.add_axes([0.5, 0.2, 0.01, 0.6])  # [x, y, width, height]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Data Value')
    
    # Make layout tight
    plt.tight_layout()
    
    def update(frame):
        """Update function for animation"""
        # Clear both axes
        ax1.clear()
        ax2.clear()
        
        # Update left plot (data1)
        surfaces1 = create_3d_box_surfaces(data1[frame])
        for x, y, z, colors, label in surfaces1:
            ax1.plot_surface(x, y, z, facecolors=cmap(norm(colors)), 
                           alpha=1.0, shade=False)
        
        # Remove axis labels, ticks, and tick labels
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_zlabel('')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_xlim(0, nx-1)
        ax1.set_ylim(0, ny-1)
        ax1.set_zlim(0, nz-1)
        ax1.set_title(f'{labels[0]}')
        
        # Update right plot (data2)
        surfaces2 = create_3d_box_surfaces(data2[frame])
        for x, y, z, colors, label in surfaces2:
            ax2.plot_surface(x, y, z, facecolors=cmap(norm(colors)), 
                           alpha=1.0, shade=False)
        
        # Remove axis labels, ticks, and tick labels
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_zlabel('')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_xlim(0, nx-1)
        ax2.set_ylim(0, ny-1)
        ax2.set_zlim(0, nz-1)
        ax2.set_title(f'{labels[1]}')
        
        # Update main title with current timestep
        timestep_title.set_text(f'Timestep: {frame} / {nt-1}')
        
        return [timestep_title]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=nt, 
                        interval=interval, blit=False)
    plt.close()  # Close the static figure
    return anim

def run_3d_box_visualization(data, interval=100, create_animation=True):
    """
    Run 3D box visualization and create animation
    
    Parameters:
    - data: 4D array of shape [nt, nx, ny, nz]
    - interval: Animation interval in milliseconds
    - create_animation: Whether to create and display animation
    
    Returns:
    - animation object (if created)
    """
    nt, nx, ny, nz = data.shape
    # print(f"Visualizing 3D box data with shape: {data.shape}")
    # print(f"Time steps: {nt}, Grid size: {nx}x{ny}x{nz}")
    
    # Create animation if requested
    anim = None
    if create_animation:
        print("Creating 3D box animation...")
        anim = create_3d_box_animation(data, interval=interval)
        display(HTML(anim.to_jshtml()))
    
    return anim



import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 200  # Set to 50MB or whatever you need
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from IPython.display import HTML

def run_side_by_side_visualization(data1, data2, labels=None, main_title=None, interval=100, 
                                   create_animation=True, save_gif=False, gif_filename=None,
                                   save_mp4=False, mp4_filename=None, fps=10):
    """
    Run side-by-side 3D box visualization and create animation
    
    Parameters:
    - data1: First 4D array of shape [nt, nx, ny, nz]
    - data2: Second 4D array of shape [nt, nx, ny, nz]
    - labels: Tuple of labels for the datasets
    - main_title: Main title for the entire figure
    - interval: Animation interval in milliseconds
    - create_animation: Whether to create and display animation
    - save_gif: Whether to save animation as GIF
    - gif_filename: Filename for the GIF (if None, auto-generates)
    - save_mp4: Whether to save animation as MP4
    - mp4_filename: Filename for the MP4 (if None, auto-generates)
    - fps: Frames per second for MP4 video
    
    Returns:
    - animation object (if created)
    """
    if labels is None:
        labels = ('Data1', 'Data2')
    
    if main_title is None:
        main_title = "3D Box Surface Comparison"
    
    # Create animation if requested
    anim = None
    if create_animation:
        print("Creating side-by-side 3D box animation...")
        anim = create_side_by_side_animation(data1, data2, labels, main_title, interval=interval)
        
        # Display the animation
        display(HTML(anim.to_jshtml()))
        
        # Save as GIF if requested
        if save_gif:
            if gif_filename is None:
                # Auto-generate filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gif_filename = f"3d_comparison_{timestamp}.gif"
            
            print(f"Saving animation as GIF: {gif_filename}")
            
            # Method 1: Using Pillow (PIL) - Recommended
            try:
                # This requires: pip install pillow
                anim.save(gif_filename, writer='pillow', fps=1000/interval)
                print(f"GIF saved successfully: {gif_filename}")
            except Exception as e:
                print(f"Error saving with Pillow: {e}")
                
                # Method 2: Using ImageMagick as fallback
                try:
                    # This requires ImageMagick to be installed
                    anim.save(gif_filename, writer='imagemagick', fps=1000/interval)
                    print(f"GIF saved successfully with ImageMagick: {gif_filename}")
                except Exception as e2:
                    print(f"Error saving with ImageMagick: {e2}")
                    print("Please install either Pillow or ImageMagick to save GIFs")
        
        # Save as MP4 if requested
        if save_mp4:
            if mp4_filename is None:
                # Auto-generate filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mp4_filename = f"3d_comparison_{timestamp}.mp4"
            
            print(f"Saving animation as MP4: {mp4_filename}")
            
            # Method 1: Using FFmpeg writer (Recommended)
            try:
                # This requires ffmpeg to be installed
                anim.save(mp4_filename, writer='ffmpeg', fps=fps, bitrate=1800)
                print(f"MP4 saved successfully: {mp4_filename}")
            except Exception as e:
                print(f"Error saving with FFmpeg: {e}")
                
                # Method 2: Using HTML5 video writer as fallback
                try:
                    anim.save(mp4_filename, writer='html5', fps=fps)
                    print(f"MP4 saved successfully with HTML5 writer: {mp4_filename}")
                except Exception as e2:
                    print(f"Error saving with HTML5 writer: {e2}")
                    print("Please install FFmpeg to save MP4 videos")
                    print("Install with: conda install ffmpeg or apt-get install ffmpeg")
    
    return anim

def plot_3d_box_snapshots(data_list, subplot_titles=None, main_title="3D Box Snapshots", 
                         figsize=None, vmin=-0.9, vmax=0.9, cmap='bwr', show_axes=False, 
                         show_colorbar=True, save_path=None):
    """
    Plot static snapshots of 3D box surfaces side by side
    
    Parameters:
    - data_list: List of 3D arrays [nx, ny, nz]. Can be 2, 3, or 4 datasets
    - subplot_titles: List of titles for each subplot. If None, uses default titles
    - main_title: Main title for the entire figure
    - figsize: Figure size (width, height). If None, auto-calculated
    - vmin, vmax: Color scale limits
    - cmap: Colormap name
    - show_axes: Whether to show axis labels and ticks
    - show_colorbar: Whether to show colorbar
    - save_path: Path to save the figure (optional)
    
    Returns:
    - fig: matplotlib figure object
    """
    
    # Convert single dataset to list
    if not isinstance(data_list, list):
        data_list = [data_list]
    
    n_plots = len(data_list)
    if n_plots > 4:
        raise ValueError("Maximum 4 datasets supported")
    
    # Handle subplot titles
    if subplot_titles is None:
        subplot_titles = [f'Dataset {i+1}' for i in range(n_plots)]
    elif len(subplot_titles) != n_plots:
        raise ValueError("Number of subplot titles must match number of datasets")
    
    # Calculate figure size
    if figsize is None:
        width = 6 * n_plots
        height = 6
        figsize = (width, height)
    
    # Create subplots
    if n_plots == 1:
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(111, projection='3d')]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': '3d'})
    elif n_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': '3d'})
    elif n_plots == 4:
        fig, axes = plt.subplots(1, 4, figsize=figsize, subplot_kw={'projection': '3d'})
        axes = axes.flatten()
    
    # Set up color normalization
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    
    # Plot each dataset
    for i, (data, title) in enumerate(zip(data_list, subplot_titles)):
        ax = axes[i]
        
        # Validate data is 3D
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D [nx, ny, nz], got {data.ndim}D")
        
        volume = data
        nx, ny, nz = volume.shape
        
        # Create and plot surfaces
        # surfaces = create_3d_box_surfaces(volume)
        
        # from matplotlib.colors import LightSource
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # cmap = cm.get_cmap(cmap)
        # ls = LightSource(azdeg=315, altdeg=45)


        # for x, y, z, colors, label in surfaces:
        #     rgb = cmap(norm(colors))
        #     shaded_rgb = ls.shade_rgb(rgb, colors, fraction=1.0)
        #     ax.plot_surface(x, y, z, facecolors=shaded_rgb, linewidth=0, antialiased=False, shade=False)
        #     # ax.plot_surface(x, y, z, facecolors=colormap(norm(colors)), alpha=1.0, shade=False)

        surfaces = create_3d_box_surfaces(volume)
        for x, y, z, colors, label in surfaces:
            ax.plot_surface(x, y, z, facecolors=colormap(norm(colors)), alpha=1.0, shade=False)
        
        # Configure axes
        if show_axes:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        
        # Set limits
        ax.set_xlim(0, nx-1)
        ax.set_ylim(0, ny-1)
        ax.set_zlim(0, nz-1)
        
        # Set subplot title
        ax.set_title(title, fontsize = 16)
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, y=1.02)
    
    # Add colorbar
    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array([])
        
        # Method 1: Manual positioning with fig.add_axes([left, bottom, width, height])
        # Values are in figure coordinates (0 to 1)
        cbar_ax = fig.add_axes([1, 0.2, 0.01, 0.6])  # [x, y, width, height]
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Data Value')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    # return fig

def plot_4d_box_snapshots(data_list, subplot_titles=None, main_title="3D Box Snapshots", 
                         figsize=None, vmin=-0.9, vmax=0.9, cmap='bwr', show_axes=False, 
                         show_colorbar=True, save_path=None):
    """
    Plot static snapshots of 3D box surfaces side by side
    
    Parameters:
    - data_list: List of 3D arrays [nx, ny, nz]. Can be 2, 3, or 4 datasets
    - subplot_titles: List of titles for each subplot. If None, uses default titles
    - main_title: Main title for the entire figure
    - figsize: Figure size (width, height). If None, auto-calculated
    - vmin, vmax: Color scale limits
    - cmap: Colormap name
    - show_axes: Whether to show axis labels and ticks
    - show_colorbar: Whether to show colorbar
    - save_path: Path to save the figure (optional)
    
    Returns:
    - fig: matplotlib figure object
    """
    
    # Convert single dataset to list
    if not isinstance(data_list, list):
        data_list = [data_list]
    
    n_plots = len(data_list)
    if n_plots > 4:
        raise ValueError("Maximum 4 datasets supported")
    
    # Handle subplot titles
    if subplot_titles is None:
        subplot_titles = [f'Dataset {i+1}' for i in range(n_plots)]
    elif len(subplot_titles) != n_plots:
        raise ValueError("Number of subplot titles must match number of datasets")
    
    # Calculate figure size
    if figsize is None:
        width = 6 * n_plots
        height = 6
        figsize = (width, height)
    
    # Create subplots
    if n_plots == 1:
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(111, projection='3d')]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': '3d'})
    elif n_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': '3d'})
    elif n_plots == 4:
        fig, axes = plt.subplots(1, 4, figsize=figsize, subplot_kw={'projection': '3d'})
        axes = axes.flatten()
    
    # Set up color normalization
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    
    # Plot each dataset
    for i, (data, title) in enumerate(zip(data_list, subplot_titles)):
        ax = axes[i]
        
        # Validate data is 3D
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D [nx, ny, nz], got {data.ndim}D")
        
        volume = data
        nx, ny, nz = volume.shape
        
        # Create and plot surfaces
        # surfaces = create_3d_box_surfaces(volume)
        
        # from matplotlib.colors import LightSource
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # cmap = cm.get_cmap(cmap)
        # ls = LightSource(azdeg=315, altdeg=45)


        # for x, y, z, colors, label in surfaces:
        #     rgb = cmap(norm(colors))
        #     shaded_rgb = ls.shade_rgb(rgb, colors, fraction=1.0)
        #     ax.plot_surface(x, y, z, facecolors=shaded_rgb, linewidth=0, antialiased=False, shade=False)
        #     # ax.plot_surface(x, y, z, facecolors=colormap(norm(colors)), alpha=1.0, shade=False)

        surfaces = create_3d_box_surfaces(volume)
        for x, y, z, colors, label in surfaces:
            ax.plot_surface(x, y, z, facecolors=colormap(norm(colors)), alpha=1.0, shade=False)
        
        # Configure axes
        if show_axes:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        
        # Set limits
        ax.set_xlim(0, nx-1)
        ax.set_ylim(0, ny-1)
        ax.set_zlim(0, nz-1)
        
        # Set subplot title
        ax.set_title(title, fontsize = 16)
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, y=1.02)
    
    # Add colorbar
    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array([])
        
        # Method 1: Manual positioning with fig.add_axes([left, bottom, width, height])
        # Values are in figure coordinates (0 to 1)
        cbar_ax = fig.add_axes([1, 0.2, 0.01, 0.6])  # [x, y, width, height]
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Data Value')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    # return fig

# Example usage functions
def plot_2_snapshots(data1, data2, titles=None, main_title="Comparison", save_path=None, vmin=0, vmax=1, cmap='bwr'):
    """Convenience function for 2 snapshots"""
    if titles is None:
        titles = ['t=0', 't=70']
    
    return plot_3d_box_snapshots(
        data_list=[data1, data2],
        subplot_titles=titles,
        main_title=main_title, save_path=save_path,
        vmin=vmin, vmax=vmax, cmap=cmap
    )

def plot_3_snapshots(data1, data2, data3,data4, titles=None, main_title="Comparison", save_path=None, vmin=0, vmax=1, cmap='bwr'):
    """Convenience function for 3 snapshots"""
    if titles is None:
        titles = ['t=0', 't=10', 't=20','t=70']
    
    return plot_3d_box_snapshots(
        data_list=[data1, data2, data3,data4],
        subplot_titles=titles,
        main_title=main_title,
        save_path=save_path,
        vmin=vmin, vmax=vmax, cmap=cmap
    )

def plot_4_snapshots(data1, data2, data3, data4, titles=None, main_title="Comparison", save_path=None, vmin=-0.9, vmax=0.9, cmap='bwr'):
    """Convenience function for 4 snapshots"""
    if titles is None:
        titles = ['Snapshot 1', 'Snapshot 2', 'Snapshot 3', 'Snapshot 4']
    
    return plot_3d_box_snapshots(
        data_list=[data1, data2, data3, data4],
        subplot_titles=titles,
        main_title=main_title,
        save_path=save_path,
        vmin=vmin, vmax=vmax, cmap=cmap
    )

def plot_single_snapshot(data, title="3D Box Snapshot", main_title="3D Visualization", show_colorbar=True, show_axes=False, save_path=None):
    """Convenience function for single snapshot"""
    return plot_3d_box_snapshots(
        data_list=[data],
        subplot_titles=[title],
        main_title=main_title, 
        show_colorbar=show_colorbar, vmin=-0.9, vmax=0.9, cmap='bwr', show_axes=show_axes, save_path=save_path
    )
    
    
    import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML

def create_histogram_animation(data, bins=50, interval=100, alpha=0.7, color='skyblue', 
                              title_prefix="Grid Values Distribution", 
                              xlabel="Value", ylabel="Frequency"):
    """
    Create an animation of histograms showing the distribution of grid point values over time
    
    Parameters:
    - data: Array of shape [nt, nx, ny, nz] - 4D data with time as first dimension
    - bins: Number of histogram bins or bin edges
    - interval: Time between frames in milliseconds
    - alpha: Transparency of histogram bars
    - color: Color of histogram bars
    - title_prefix: Prefix for the title (timestep will be appended)
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    
    Returns:
    - Animation object
    """
    nt = data.shape[0]
    
    # Calculate global min/max for consistent x-axis
    global_min = data.min()
    global_max = data.max()
    
    # If bins is an integer, create bin edges
    if isinstance(bins, int):
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        bin_edges = bins
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate initial histogram
    flat_data = data[0].flatten()
    counts, _ = np.histogram(flat_data, bins=bin_edges)
    
    # Create initial bar plot
    bars = ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), 
                  alpha=alpha, color=color, edgecolor='black', linewidth=0.5)
    
    # Set up axes
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(global_min, global_max)
    
    # Calculate global max count for consistent y-axis
    max_count = 0
    for t in range(nt):
        flat_data_t = data[t].flatten()
        counts_t, _ = np.histogram(flat_data_t, bins=bin_edges)
        max_count = max(max_count, counts_t.max())
    
    ax.set_ylim(0, max_count * 1.1)
    
    # Set initial title
    title = ax.set_title(f'{title_prefix} - Timestep: 0/{nt-1}', fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    def update(frame):
        """Update function for animation"""
        # Calculate histogram for current timestep
        flat_data = data[frame].flatten()
        counts, _ = np.histogram(flat_data, bins=bin_edges)
        
        # Update bar heights
        for bar, count in zip(bars, counts):
            bar.set_height(count)
        
        # Update title
        title.set_text(f'{title_prefix} - Timestep: {frame}/{nt-1}')
        
        return bars + [title]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=nt, interval=interval, blit=False)
    plt.close()  # Close the static figure
    return anim


def create_side_by_side_histogram_animation(data1, data2, bins=50, interval=100, 
                                          colors=['skyblue', 'lightcoral'],
                                          labels=None, main_title=None, 
                                          xlabel="Value", ylabel="Frequency"):
    """
    Create a side-by-side histogram animation comparing two datasets
    
    Parameters:
    - data1: First dataset of shape [nt, nx, ny, nz]
    - data2: Second dataset of shape [nt, nx, ny, nz]
    - bins: Number of histogram bins or bin edges
    - interval: Time between frames in milliseconds
    - colors: List of colors for the two histograms
    - labels: Tuple of labels for the two datasets
    - main_title: Main title for the entire figure
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    
    Returns:
    - Animation object
    """
    if labels is None:
        labels = ('Data1', 'Data2')
    
    if main_title is None:
        main_title = "Grid Values Distribution Comparison"
    
    # Ensure both datasets have the same shape
    assert data1.shape == data2.shape, "Both datasets must have the same shape"
    
    nt = data1.shape[0]
    
    # Calculate global min/max for consistent x-axis across both datasets
    global_min = min(data1.min(), data2.min())
    global_max = max(data1.max(), data2.max())
    
    # If bins is an integer, create bin edges
    if isinstance(bins, int):
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        bin_edges = bins
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate global max count for consistent y-axis
    max_count = 0
    for t in range(nt):
        for data in [data1, data2]:
            flat_data = data[t].flatten()
            counts, _ = np.histogram(flat_data, bins=bin_edges)
            max_count = max(max_count, counts.max())
    max_count=500
    # Initialize first subplot
    flat_data1 = data1[0].flatten()
    counts1, _ = np.histogram(flat_data1, bins=bin_edges)
    bars1 = ax1.bar(bin_edges[:-1], counts1, width=np.diff(bin_edges), 
                    alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_xlim(global_min, global_max)
    ax1.set_ylim(0, max_count * 1.1)
    ax1.set_title(labels[0], fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Initialize second subplot
    flat_data2 = data2[0].flatten()
    counts2, _ = np.histogram(flat_data2, bins=bin_edges)
    bars2 = ax2.bar(bin_edges[:-1], counts2, width=np.diff(bin_edges), 
                    alpha=0.7, color=colors[1], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_xlim(global_min, global_max)
    ax2.set_ylim(0, max_count * 1.1)
    ax2.set_title(labels[1], fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add main title and timestep title
    fig.suptitle(main_title, fontsize=16, y=0.98)
    timestep_title = fig.text(0.5, 0.02, f'Timestep: 0 / {nt-1}', 
                             fontsize=12, ha='center', transform=fig.transFigure)
    
    plt.tight_layout()
    
    def update(frame):
        """Update function for animation"""
        # Update first histogram
        flat_data1 = data1[frame].flatten()
        counts1, _ = np.histogram(flat_data1, bins=bin_edges)
        for bar, count in zip(bars1, counts1):
            bar.set_height(count)
        
        # Update second histogram
        flat_data2 = data2[frame].flatten()
        counts2, _ = np.histogram(flat_data2, bins=bin_edges)
        for bar, count in zip(bars2, counts2):
            bar.set_height(count)
        
        # Update timestep title
        timestep_title.set_text(f'Timestep: {frame} / {nt-1}')
        
        return list(bars1) + list(bars2) + [timestep_title]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=nt, interval=interval, blit=False)
    plt.close()  # Close the static figure
    return anim


def create_overlay_histogram_animation(data1, data2, bins=50, interval=100, 
                                     colors=['skyblue', 'lightcoral'], 
                                     labels=None, main_title=None,
                                     alpha=0.6, xlabel="Value", ylabel="Frequency"):
    """
    Create an overlaid histogram animation comparing two datasets on the same plot
    
    Parameters:
    - data1: First dataset of shape [nt, nx, ny, nz]
    - data2: Second dataset of shape [nt, nx, ny, nz]
    - bins: Number of histogram bins or bin edges
    - interval: Time between frames in milliseconds
    - colors: List of colors for the two histograms
    - labels: Tuple of labels for the two datasets
    - main_title: Main title for the figure
    - alpha: Transparency of histogram bars
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    
    Returns:
    - Animation object
    """
    if labels is None:
        labels = ('Data1', 'Data2')
    
    if main_title is None:
        main_title = "Grid Values Distribution Overlay"
    
    # Ensure both datasets have the same shape
    assert data1.shape == data2.shape, "Both datasets must have the same shape"
    
    nt = data1.shape[0]
    
    # Calculate global min/max for consistent x-axis
    global_min = min(data1.min(), data2.min())
    global_max = max(data1.max(), data2.max())
    
    # If bins is an integer, create bin edges
    if isinstance(bins, int):
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        bin_edges = bins
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate global max count for consistent y-axis
    max_count = 0
    for t in range(nt):
        for data in [data1, data2]:
            flat_data = data[t].flatten()
            counts, _ = np.histogram(flat_data, bins=bin_edges)
            max_count = max(max_count, counts.max())
    
    # Initialize histograms
    flat_data1 = data1[0].flatten()
    flat_data2 = data2[0].flatten()
    counts1, _ = np.histogram(flat_data1, bins=bin_edges)
    counts2, _ = np.histogram(flat_data2, bins=bin_edges)
    
    # Create bars with slight offset for better visibility
    width = np.diff(bin_edges) * 0.8
    offset = width * 0.1
    
    bars1 = ax.bar(bin_edges[:-1] - offset/2, counts1, width=width/2, 
                   alpha=alpha, color=colors[0], label=labels[0], 
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(bin_edges[:-1] + offset/2, counts2, width=width/2, 
                   alpha=alpha, color=colors[1], label=labels[1], 
                   edgecolor='black', linewidth=0.5)
    
    # Set up axes
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(0, max_count * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set title
    title = ax.set_title(f'{main_title} - Timestep: 0/{nt-1}', fontsize=14)
    
    def update(frame):
        """Update function for animation"""
        # Update first histogram
        flat_data1 = data1[frame].flatten()
        counts1, _ = np.histogram(flat_data1, bins=bin_edges)
        for bar, count in zip(bars1, counts1):
            bar.set_height(count)
        
        # Update second histogram
        flat_data2 = data2[frame].flatten()
        counts2, _ = np.histogram(flat_data2, bins=bin_edges)
        for bar, count in zip(bars2, counts2):
            bar.set_height(count)
        
        # Update title
        title.set_text(f'{main_title} - Timestep: {frame}/{nt-1}')
        
        return list(bars1) + list(bars2) + [title]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=nt, interval=interval, blit=False)
    plt.close()  # Close the static figure
    return anim


def run_histogram_visualization(data, bins=50, interval=100, create_animation=True, 
                               title_prefix="Grid Values Distribution"):
    """
    Run histogram visualization and create animation
    
    Parameters:
    - data: 4D array of shape [nt, nx, ny, nz]
    - bins: Number of histogram bins
    - interval: Animation interval in milliseconds
    - create_animation: Whether to create and display animation
    - title_prefix: Prefix for the animation title
    
    Returns:
    - animation object (if created)
    """
    nt, nx, ny, nz = data.shape
    print(f"Visualizing histogram data with shape: {data.shape}")
    print(f"Time steps: {nt}, Grid size: {nx}x{ny}x{nz}")
    print(f"Total grid points per timestep: {nx*ny*nz}")
    
    # Create animation if requested
    anim = None
    if create_animation:
        print("Creating histogram animation...")
        anim = create_histogram_animation(data, bins=bins, interval=interval, 
                                        title_prefix=title_prefix)
        display(HTML(anim.to_jshtml()))
    
    return anim


def run_side_by_side_histogram_visualization(data1, data2, bins=50, interval=100, 
                                           labels=None, main_title=None, 
                                           create_animation=True):
    """
    Run side-by-side histogram visualization and create animation
    
    Parameters:
    - data1: First 4D array of shape [nt, nx, ny, nz]
    - data2: Second 4D array of shape [nt, nx, ny, nz]
    - bins: Number of histogram bins
    - interval: Animation interval in milliseconds
    - labels: Tuple of labels for the datasets
    - main_title: Main title for the entire figure
    - create_animation: Whether to create and display animation
    
    Returns:
    - animation object (if created)
    """
    if labels is None:
        labels = ('Data1', 'Data2')
    
    if main_title is None:
        main_title = "Grid Values Distribution Comparison"
    
    nt, nx, ny, nz = data1.shape
    print(f"Visualizing side-by-side histogram data with shape: {data1.shape}")
    print(f"Time steps: {nt}, Grid size: {nx}x{ny}x{nz}")
    print(f"Comparing: {labels[0]} vs {labels[1]}")
    
    # Create animation if requested
    anim = None
    if create_animation:
        print("Creating side-by-side histogram animation...")
        anim = create_side_by_side_histogram_animation(data1, data2, bins=bins, 
                                                     interval=interval, labels=labels, 
                                                     main_title=main_title)
        display(HTML(anim.to_jshtml()))
    
    return anim


def run_overlay_histogram_visualization(data1, data2, bins=50, interval=100, 
                                      labels=None, main_title=None, 
                                      create_animation=True):
    """
    Run overlaid histogram visualization and create animation
    
    Parameters:
    - data1: First 4D array of shape [nt, nx, ny, nz]
    - data2: Second 4D array of shape [nt, nx, ny, nz]
    - bins: Number of histogram bins
    - interval: Animation interval in milliseconds
    - labels: Tuple of labels for the datasets
    - main_title: Main title for the figure
    - create_animation: Whether to create and display animation
    
    Returns:
    - animation object (if created)
    """
    if labels is None:
        labels = ('Data1', 'Data2')
    
    if main_title is None:
        main_title = "Grid Values Distribution Overlay"
    
    nt, nx, ny, nz = data1.shape
    print(f"Visualizing overlaid histogram data with shape: {data1.shape}")
    print(f"Time steps: {nt}, Grid size: {nx}x{ny}x{nz}")
    print(f"Comparing: {labels[0]} vs {labels[1]}")
    
    # Create animation if requested
    anim = None
    if create_animation:
        print("Creating overlaid histogram animation...")
        anim = create_overlay_histogram_animation(data1, data2, bins=bins, 
                                                interval=interval, labels=labels, 
                                                main_title=main_title)
        display(HTML(anim.to_jshtml()))
    
    return anim








def plot_single_histogram(data, bins=50, title=None, figsize=None, 
                          color='black', alpha=0.5, ylabel='Number of voxels',
                          xlabel='Concentration value c', show_grid=True, save_path=None,
                          xlim=None, ylim=None, show_stats=True, 
                          stats_position=(0.02, 0.98), yticks=None, xticks=None,
                          data2=None, line_color='blue', line_width=1, 
                          line_style='-', line_label='Line Data',
                          bar_label='GT', line_legend_label='ML',
                          data3=None, line_color2='red', line_width2=1,
                          line_style2='--', line_legend_label2='ML2', 
                          data4=None, bar_color2='green', bar_alpha2=0.5,
                          bar_label2='GT2', nsample=1):
    """
    Plot a single histogram with up to four datasets, normalized by number of samples
    
    Parameters:
    - data: Array of data (can be 3D array [nx, ny, nz] or 1D array) or list of arrays for multiple samples - plotted as bars
    - bins: Number of histogram bins, bin edges, or tuple (min, max, n_bins)
    - title: Title for the histogram
    - figsize: Figure size (width, height). If None, uses (10, 6)
    - color: Color for the histogram bars
    - alpha: Transparency of histogram bars
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - show_grid: Whether to show grid lines
    - save_path: Path to save the figure (optional)
    - xlim: Tuple (xmin, xmax) for x-axis limits. If None, uses data range
    - ylim: Tuple (ymin, ymax) for y-axis limits. If None, auto-calculated
    - show_stats: Whether to display mean and std statistics on the plot
    - stats_position: Tuple (x, y) for statistics text position in axes coordinates (0-1)
    - yticks: List/array of y-tick values. If None, uses automatic ticks
    - xticks: List/array of x-tick values. If None, uses automatic ticks
    - data2: Second dataset to plot as line histogram (optional) - can be list of arrays for multiple samples
    - line_color: Color for the first line histogram
    - line_width: Width of the first line
    - line_style: Style of the first line ('-', '--', '-.', ':')
    - line_label: Label for the first line in legend
    - bar_label: Label name for the bar data in legend (default: 'GT')
    - line_legend_label: Label name for the first line data in legend (default: 'ML')
    - data3: Third dataset to plot as second line histogram (optional) - can be list of arrays for multiple samples
    - line_color2: Color for the second line histogram
    - line_width2: Width of the second line
    - line_style2: Style of the second line ('-', '--', '-.', ':')
    - line_legend_label2: Label name for the second line data in legend (default: 'ML2')
    - data4: Fourth dataset to plot as second bar histogram (optional) - can be list of arrays for multiple samples
    - bar_color2: Color for the second bar histogram (default: 'green')
    - bar_alpha2: Transparency for the second bar histogram (default: 0.5)
    - bar_label2: Label name for the second bar data in legend (default: 'GT2')
    - nsample: Number of samples for normalization (counts will be divided by this number)
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    
    # Calculate figure size
    if figsize is None:
        figsize = (10, 6)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Helper function to process data (handle both single arrays and lists of arrays)
    def process_data(input_data):
        if input_data is None:
            return None, None
        
        # Check if input_data is a list of arrays (multiple samples)
        if isinstance(input_data, list):
            # Concatenate all samples
            all_data = []
            for sample in input_data:
                if sample.ndim > 1:
                    all_data.extend(sample.flatten())
                else:
                    all_data.extend(sample)
            flat_data = np.array(all_data)
            actual_nsamples = len(input_data)
        else:
            # Single array
            if input_data.ndim > 1:
                flat_data = input_data.flatten()
            else:
                flat_data = input_data
            actual_nsamples = 1
        
        return flat_data, actual_nsamples
    
    # Process all datasets
    flat_data, nsamples_data1 = process_data(data)
    flat_data2, nsamples_data2 = process_data(data2)
    flat_data3, nsamples_data3 = process_data(data3)
    flat_data4, nsamples_data4 = process_data(data4)
    
    # Update nsample based on actual data if not provided correctly
    if isinstance(data, list):
        nsample = max(nsample, nsamples_data1)
    
    # Handle x-axis limits
    if xlim is None:
        data_min = flat_data.min()
        data_max = flat_data.max()
        if flat_data2 is not None:
            data_min = min(data_min, flat_data2.min())
            data_max = max(data_max, flat_data2.max())
        if flat_data3 is not None:
            data_min = min(data_min, flat_data3.min())
            data_max = max(data_max, flat_data3.max())
        if flat_data4 is not None:
            data_min = min(data_min, flat_data4.min())
            data_max = max(data_max, flat_data4.max())
        xlim = (data_min, data_max)
    else:
        data_min, data_max = xlim
    
    # Handle bins
    if isinstance(bins, tuple) and len(bins) == 3:
        # bins is (min, max, n_bins)
        bin_min, bin_max, n_bins = bins
        bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)
    elif isinstance(bins, int):
        # bins is number of bins, use data range
        bin_edges = np.linspace(data_min, data_max, bins + 1)
    else:
        # bins is array of bin edges
        bin_edges = bins
    
    # Calculate bin properties
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_width = np.diff(bin_edges)
    
    # Create histogram for first dataset (GT) - always as bars
    counts, _ = np.histogram(flat_data, bins=bin_edges)
    counts_normalized = counts / nsample  # Normalize by number of samples
    ax.bar(bin_centers, counts_normalized, width=bin_width, 
           alpha=alpha, color=color, edgecolor='black', linewidth=0.5,
           label=bar_label)
    
    # Create line histogram for GT2 dataset if provided (normalized)
    if flat_data4 is not None:
        counts4, _ = np.histogram(flat_data4, bins=bin_edges)
        counts4_normalized = counts4 / nsample  # Normalize by number of samples
        ax.plot(bin_centers, counts4_normalized, color=bar_color2, linewidth=line_width, 
                linestyle='-', label=bar_label2, marker='d', markersize=3)
    
    # Create line histogram for second dataset if provided (normalized)
    if flat_data2 is not None:
        counts2, _ = np.histogram(flat_data2, bins=bin_edges)
        counts2_normalized = counts2 / nsample  # Normalize by number of samples
        ax.plot(bin_centers, counts2_normalized, color=line_color, linewidth=line_width, 
                linestyle=line_style, label=line_legend_label, marker='o', markersize=2)
    
    # Create line histogram for third dataset if provided (normalized)
    if flat_data3 is not None:
        counts3, _ = np.histogram(flat_data3, bins=bin_edges)
        counts3_normalized = counts3 / nsample  # Normalize by number of samples
        ax.plot(bin_centers, counts3_normalized, color=line_color2, linewidth=line_width2, 
                linestyle=line_style2, label=line_legend_label2, marker='s', markersize=2)
    
    # Configure axes
    ax.set_xlabel(xlabel, fontsize=12)
    # Update ylabel to reflect normalization
    if nsample > 1:
        ylabel_normalized = f'{ylabel}'
    else:
        ylabel_normalized = ylabel
    ax.set_ylabel(ylabel_normalized, fontsize=12)
    ax.set_xlim(xlim)
    
    # Set y-axis limits
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        max_count = counts_normalized.max()
        if flat_data2 is not None:
            counts2_normalized = counts2 / nsample
            max_count = max(max_count, counts2_normalized.max())
        if flat_data3 is not None:
            counts3_normalized = counts3 / nsample
            max_count = max(max_count, counts3_normalized.max())
        if flat_data4 is not None:
            counts4_normalized = counts4 / nsample
            max_count = max(max_count, counts4_normalized.max())
        ax.set_ylim(0, max_count * 1.1)
    
    ax.set_title(title, fontsize=14)
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Set custom y-ticks if provided
    if yticks is not None:
        ax.set_yticks(yticks)
    
    # Set custom x-ticks if provided
    if xticks is not None:
        ax.set_xticks(xticks)
    
    # Add statistics text or legend at stats_position
    if show_stats:
        mean_val = np.mean(flat_data)
        std_val = np.std(flat_data)
        
        if flat_data2 is not None or flat_data3 is not None or flat_data4 is not None:
            # When multiple datasets: create custom legend with statistics and visual indicators
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Patch(facecolor=color, alpha=alpha, 
                      label=f'{bar_label} (μ={mean_val:.2f}, σ={std_val:.3f})')
            ]
            
            if flat_data4 is not None:
                mean_val4 = np.mean(flat_data4)
                std_val4 = np.std(flat_data4)
                legend_elements.append(
                    Line2D([0], [0], color=bar_color2, linewidth=line_width, 
                           linestyle='-', marker='d', markersize=2,
                           label=f'{bar_label2} (μ={mean_val4:.2f}, σ={std_val4:.3f})')
                )
            
            if flat_data2 is not None:
                mean_val2 = np.mean(flat_data2)
                std_val2 = np.std(flat_data2)
                legend_elements.append(
                    Line2D([0], [0], color=line_color, linewidth=line_width, 
                           linestyle=line_style, marker='o', markersize=2,
                           label=f'{line_legend_label} (μ={mean_val2:.2f}, σ={std_val2:.3f})')
                )
            
            if flat_data3 is not None:
                mean_val3 = np.mean(flat_data3)
                std_val3 = np.std(flat_data3)
                legend_elements.append(
                    Line2D([0], [0], color=line_color2, linewidth=line_width2, 
                           linestyle=line_style2, marker='s', markersize=2,
                           label=f'{line_legend_label2} (μ={mean_val3:.2f}, σ={std_val3:.3f})')
                )
            
            # Create legend at the specified position
            legend = ax.legend(handles=legend_elements, loc='center', 
                             bbox_to_anchor=stats_position, frameon=True, 
                             fancybox=True, shadow=False, framealpha=0.8,
                             facecolor='white', edgecolor='black')
            legend.get_frame().set_boxstyle('round,pad=0.5')
            
        else:
            # Single dataset: show statistics as before
            stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nSamples: {nsample}'
            ax.text(stats_position[0], stats_position[1], stats_text, 
                    transform=ax.transAxes, horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, ax









    

def plot_3_snapshots_histogram(data1, data2, data3, bins=50, titles=None, 
                               main_title="Histogram Comparison", figsize=None, 
                               colors=['skyblue', 'lightcoral', 'lightgreen'],
                               alpha=0.7, xlabel="Value", ylabel="Frequency", 
                               show_grid=True, save_path=None,
                               xlim=None, ylim=None, use_global_ylim=True):
    """
    Plot static histograms of 3 datasets side by side
    
    Parameters:
    - data1, data2, data3: 3D arrays [nx, ny, nz] or flattened 1D arrays
    - bins: Number of histogram bins, bin edges, or tuple (min, max, n_bins)
    - titles: List of titles for each subplot. If None, uses default titles
    - main_title: Main title for the entire figure
    - figsize: Figure size (width, height). If None, auto-calculated
    - colors: List of colors for the three histograms
    - alpha: Transparency of histogram bars
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - show_grid: Whether to show grid lines
    - save_path: Path to save the figure (optional)
    - xlim: Tuple (xmin, xmax) for x-axis limits. If None, uses data range
    - ylim: Tuple (ymin, ymax) for y-axis limits. If None, auto-calculated
    - use_global_ylim: If True and ylim is None, use same y-limits for all subplots
    
    Returns:
    - fig: matplotlib figure object
    """
    
    # Handle subplot titles
    if titles is None:
        titles = ['Histogram 1', 'Histogram 2', 'Histogram 3']
    elif len(titles) != 3:
        raise ValueError("Number of subplot titles must be 3")
    
    # Calculate figure size
    if figsize is None:
        figsize = (18, 6)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Flatten data if needed
    datasets = []
    for data in [data1, data2, data3]:
        if data.ndim > 1:
            flat_data = data.flatten()
        else:
            flat_data = data
        datasets.append(flat_data)
    
    # Handle x-axis limits
    if xlim is None:
        # Calculate global min/max for consistent x-axis from data
        global_min = min(data.min() for data in datasets)
        global_max = max(data.max() for data in datasets)
        xlim = (global_min, global_max)
    else:
        # Use provided x-axis limits
        global_min, global_max = xlim
    
    # Handle bins
    if isinstance(bins, tuple) and len(bins) == 3:
        # bins is (min, max, n_bins)
        bin_min, bin_max, n_bins = bins
        bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)
    elif isinstance(bins, int):
        # bins is number of bins, use xlim range
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        # bins is array of bin edges
        bin_edges = bins
    
    # Calculate y-axis limits
    if ylim is None:
        if use_global_ylim:
            # Calculate global max count for consistent y-axis
            max_count = 0
            for data in datasets:
                counts, _ = np.histogram(data, bins=bin_edges)
                max_count = max(max_count, counts.max())
            ylim = (0, max_count * 1.1)
        else:
            # Will be calculated individually for each subplot
            ylim = None
    
    # Plot each histogram
    for i, (data, title, color) in enumerate(zip(datasets, titles, colors)):
        ax = axes[i]
        
        # Create histogram
        counts, _ = np.histogram(data, bins=bin_edges)
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), 
               alpha=alpha, color=color, edgecolor='black', linewidth=0.5)
        
        # Configure axes
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim)
        
        # Set y-axis limits
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            # Auto-calculate for this subplot
            ax.set_ylim(0, counts.max() * 1.1)
        
        ax.set_title(title, fontsize=14)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_4_snapshots_histogram(data1, data2, data3, data4, bins=50, titles=None, 
                               main_title="Histogram Comparison", figsize=None, 
                               colors=['blue', 'orange', 'green', 'red'],
                               alpha=0.7, xlabel="Value", ylabel="Frequency", 
                               show_grid=True, save_path=None,
                               xlim=None, ylim=None, use_global_ylim=True):
    """
    Plot static histograms of 4 datasets side by side
    
    Parameters:
    - data1, data2, data3, data4: 3D arrays [nx, ny, nz] or flattened 1D arrays
    - bins: Number of histogram bins, bin edges, or tuple (min, max, n_bins)
    - titles: List of titles for each subplot. If None, uses default titles
    - main_title: Main title for the entire figure
    - figsize: Figure size (width, height). If None, auto-calculated
    - colors: List of colors for the four histograms
    - alpha: Transparency of histogram bars
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - show_grid: Whether to show grid lines
    - save_path: Path to save the figure (optional)
    - xlim: Tuple (xmin, xmax) for x-axis limits. If None, uses data range
    - ylim: Can be:
            - None: auto-calculated
            - Tuple (ymin, ymax): same limits for all plots
            - List of 4 values [ymax1, ymax2, ymax3, ymax4]: individual ymax for each plot (ymin=0)
    - use_global_ylim: If True and ylim is None, use same y-limits for all subplots
    
    Returns:
    - fig: matplotlib figure object
    """
    
    # Handle subplot titles
    if titles is None:
        titles = ['Histogram 1', 'Histogram 2', 'Histogram 3', 'Histogram 4']
    elif len(titles) != 4:
        raise ValueError("Number of subplot titles must be 4")
    
    # Calculate figure size
    if figsize is None:
        figsize = (24, 3.5)  # Wider to accommodate 4 subplots
    
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Flatten data if needed
    datasets = []
    for data in [data1, data2, data3, data4]:
        if data.ndim > 1:
            flat_data = data.flatten()
        else:
            flat_data = data
        datasets.append(flat_data)
    
    # Handle x-axis limits
    if xlim is None:
        # Calculate global min/max for consistent x-axis from data
        global_min = min(data.min() for data in datasets)
        global_max = max(data.max() for data in datasets)
        xlim = (global_min, global_max)
    else:
        # Use provided x-axis limits
        global_min, global_max = xlim
    
    # Handle bins
    if isinstance(bins, tuple) and len(bins) == 3:
        # bins is (min, max, n_bins)
        bin_min, bin_max, n_bins = bins
        bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)
    elif isinstance(bins, int):
        # bins is number of bins, use xlim range
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        # bins is array of bin edges
        bin_edges = bins
    
    # Calculate y-axis limits
    individual_ylims = None
    if ylim is None:
        if use_global_ylim:
            # Calculate global max count for consistent y-axis
            max_count = 0
            for data in datasets:
                counts, _ = np.histogram(data, bins=bin_edges)
                max_count = max(max_count, counts.max())
            ylim = (0, max_count * 1.1)
        else:
            # Will be calculated individually for each subplot
            ylim = None
    elif isinstance(ylim, (list, tuple)) and len(ylim) == 4:
        # Individual ymax values for each plot (ymin=0)
        individual_ylims = [(0, ymax) for ymax in ylim]
        ylim = None  # Don't use global ylim
    elif isinstance(ylim, (list, tuple)) and len(ylim) == 2:
        # Global ylim tuple (ymin, ymax)
        ylim = tuple(ylim)
    
    # Plot each histogram
    for i, (data, title, color) in enumerate(zip(datasets, titles, colors)):
        ax = axes[i]
        
        # Create histogram
        counts, _ = np.histogram(data, bins=bin_edges)
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), 
               alpha=alpha, color=color, edgecolor='black', linewidth=0.5)
        
        # Configure axes
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim)
        
        # Set y-axis limits
        if individual_ylims is not None:
            # Use individual ylim for this subplot
            ax.set_ylim(individual_ylims[i])
        elif ylim is not None:
            # Use global ylim
            ax.set_ylim(ylim)
        else:
            # Auto-calculate for this subplot
            ax.set_ylim(0, counts.max() * 1.1)
        
        ax.set_title(title, fontsize=14)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
