from typing import Tuple

def map_1d_to_3d(coord: int, x_dim: int, y_dim: int, z_dim: int) -> Tuple[int, int, int]:
    """
    Map a 1D coordinate to a 3D coordinate.
    
    Args:
        coord (int): The 1D coordinate.
        x_dim (int): The size of the x dimension.
        y_dim (int): The size of the y dimension.
        z_dim (int): The size of the z dimension.
    
    Returns:
        tuple: A tuple containing the x, y, and z coordinates.
    """
    x = coord // (z_dim * y_dim)
    coord -= x * (z_dim * y_dim)
    y = coord // z_dim
    z = coord - (y * z_dim)
    return x, y, z
