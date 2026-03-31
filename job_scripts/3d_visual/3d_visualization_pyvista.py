
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU device to use

import sys
# sys.path.append('../')
from utility_plots import *


import numpy as np

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
                           surface_level: float = -0.8) -> bool:
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
            grid = numpy_to_pyvista_grid(data_3d, visualization_type, verbose=verbose)
        else:
            grid = numpy_to_pyvista_grid(data_3d, visualization_type, add_padding=False, verbose=verbose)
        
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
                lighting=False,  # Add this line
            )
            
        elif visualization_type == 'surface':
            # Surface rendering with multiple isosurfaces
            if verbose:
                print("  Creating surface rendering...")
            
            # Create multiple isosurfaces
            # iso_values = np.asarray([-0.8, 0.8])  # Example isosurface values
            iso_values = np.asarray([surface_level])  # Example isosurface values
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


gt = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')

visualize_surface_field(gt[0,0,:,:,:,0],
                        visualization_type='surface',
                        output_image='demo.png',
                        clim=(-0.6,0.6),
                        zoom_size=0.9,
                        surface_level=-0.6
                        )