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

def numpy_to_pyvista_grid(data_3d: np.ndarray, 
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
    grid.point_data['ScalarField'] = data_3d.flatten(order='F')
    
    return grid

def visualize_surface_field(data_3d: np.ndarray,
                           visualization_type: str = 'surface',  # 'volume', 'slice', 'surface'
                           opacity_mapping: Optional[list] = None,
                           color_mapping: Optional[list] = None,
                           output_image: Optional[str] = None,
                           title: str = "3D Surface Field Visualization",
                           colormap: str = 'bwr',
                           background_color: Tuple[float, float, float] = (1., 1., 1.),
                           window_size: Tuple[int, int] = (1200, 1200),
                           auto_camera: bool = True,
                           show_scalar_bar: bool = True,
                           clim: Optional[Tuple[float, float]] = None,
                           verbose: bool = False) -> bool:
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
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        if verbose:
            print(f"📊 Processing numpy data of shape: {data_3d.shape}")
            print(f"  Data range: {data_3d.min():.3f} to {data_3d.max():.3f}")
        
        # Convert to PyVista grid
        if visualization_type=='surface':
            grid = numpy_to_pyvista_grid(data_3d, verbose=verbose)
        else:
            grid = numpy_to_pyvista_grid(data_3d, add_padding=False, verbose=verbose)
        
        if verbose:
            print(f"  Grid created: {grid.n_points:,} points, {grid.n_cells:,} cells")
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.background_color = background_color
        
        # Set color limits if not provided
        if clim is None:
            clim = (-0.8, 0.8)
        
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
                    'label_font_size': 18,
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
            
            # Create orthogonal slices
            # slices = grid.slice_orthogonal()
            # Create slices at front faces
            # Create slices at front faces
            bounds = grid.bounds
            print("  Creating front slices...")
            print(f"  Grid bounds: {bounds}")
            slices = grid.slice(normal='x', origin=[bounds[1], 0, 0])  # Front X face
            slices += grid.slice(normal='y', origin=[0, bounds[3], 0])  # Front Y face
            slices += grid.slice(normal='z', origin=[0, 0, bounds[5]])  # Front Z face
            
            plotter.add_mesh(
                slices,
                # scalars='ScalarField',
                cmap=colormap,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={
                    'title': None,
                    'label_font_size': 18,
                    'title_font_size': 14,
                    'fmt': '%.1f',
                    'width': 0.3,
                    'height': 0.1,
                    'position_x': 0.62,
                    'position_y': 0.05,
                    'color': 'white' if background_color[0] < 0.5 else 'black'
                },
                smooth_shading=False,
                lighting=False,  # Add this line
            )
            
        elif visualization_type == 'surface':
            # Surface rendering with multiple isosurfaces
            if verbose:
                print("  Creating surface rendering...")
            
            # Create multiple isosurfaces
            iso_values = np.asarray([-0.8, 0.8])  # Example isosurface values
            iso_values = np.asarray([-0.8])  # Example isosurface values
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
                        show_scalar_bar=(show_scalar_bar and i == 0),  # Only show bar once
                        scalar_bar_args={
                            'title': 'Field Value',
                            'label_font_size': 18,
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
            if visualization_type == 'slice':
                data_center = (np.array(data_3d.shape)-1)/2
                data_bounds = (0,data_3d.shape[0]-1, 0, data_3d.shape[0]-1, 0, data_3d.shape[0]-1)  # Example bounds
                data_size = data_3d.shape[0]-1
            else:
                data_center = (np.array(data_3d.shape))/2
                data_bounds = (1,data_3d.shape[0]+0.2, 1, data_3d.shape[0]+0.2, 1, data_3d.shape[0]+0.2)  # Example bounds
                data_size = data_3d.shape[0]
                
            
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
            plotter.camera.zoom(1.0)
            
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

def load_and_visualize_surface_field(npy_file: str,
                                    data_index: Tuple = (0, -1, ...),
                                    visualization_type: str = 'surface',
                                    output_image: Optional[str] = None,
                                    title: Optional[str] = None,
                                    clim: Optional[Tuple[float, float]] = None,
                                    verbose: bool = False) -> bool:
    """
    Load NPY file and create surface field visualization
    
    Args:
        npy_file: Path to .npy file
        data_index: Index to extract from loaded data
        visualization_type: Type of visualization ('volume', 'slice', 'surface')
        output_image: Output image path
        title: Visualization title
        clim: Color limits as (min, max)
        verbose: Print debug information
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Load the numpy data
        data = np.load(npy_file)
        
        if verbose:
            print(f"📁 Loaded {npy_file}")
            print(f"  Original shape: {data.shape}")
        
        # Extract the specific data slice
        if len(data.shape) > 3:
            # Handle multi-dimensional data (batch, time, x, y, z, channels)
            data_3d = data[data_index[0], data_index[1], ..., 0]  # Extract first channel
        else:
            data_3d = data
        
        if verbose:
            print(f"  Extracted 3D data shape: {data_3d.shape}")
            print(f"  Data range: {data_3d.min():.3f} to {data_3d.max():.3f}")
        
        # Visualize
        return visualize_surface_field(
            data_3d=data_3d,
            visualization_type=visualization_type,
            output_image=output_image,
            title=title,
            clim=clim,
            verbose=verbose
        )
        
    except Exception as e:
        if verbose:
            print(f"✗ Error loading {npy_file}: {str(e)}")
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