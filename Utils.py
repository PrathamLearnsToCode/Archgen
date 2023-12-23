import open3d as o3d
import numpy as np
import os
from pyvista import read
import binvox


#Visualize a single 3D object
def point_cloud():

    pcd = o3d.io.read_point_cloud("/Users/pratham/Desktop/Dataset 2/T2/y1.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods = {}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i] = mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

def mesh():
    pcd = o3d.io.read_point_cloud('/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/input/y_gt.ply')
    print(pcd)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, linear_fit=False)[0]
    o3d.io.write_triangle_mesh("/Users/pratham/Desktop/" + "p_mesh_c.ply", poisson_mesh)
    my_lods = lod_mesh_export(poisson_mesh, [3000, 5000, 130000], ".ply", "/Users/pratham/Desktop/")

    input_file = "/Users/pratham/Desktop/lod_130000.ply"
    pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([pcd])



def voxelization(input_file, output_file, voxel_size):
    # Read the PLY file using PyVista
    mesh = read(input_file)

    # Extract vertices
    vertices = mesh.points

    # Determine the bounding box of the mesh
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)

    # Calculate voxel dimensions
    voxel_dimensions = ((max_bounds - min_bounds) / voxel_size).astype(int) + 1

    # Create a binvox instance
    binvox = binvox.Voxels(
        np.zeros(voxel_dimensions, dtype=bool),
        dims=voxel_dimensions,
        translate=min_bounds,
        scale=voxel_size,
        axis_order='xyz'
    )

    # Voxelization by marking occupied voxels
    for vertex in vertices:
        voxel_coords = ((vertex - min_bounds) / voxel_size).astype(int)
        binvox.data[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = True

    # Save the voxelized data in binvox format
    with open(output_file, 'wb') as f:
        binvox.write(f)

def main():
    input_directory = '/Users/pratham/Desktop/Dataset 2/T1'
    output_directory = '/Users/pratham/Desktop/3D reconstruction/Voxelized'
    voxel_size = 0.01  # Adjust as needed

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.ply'):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, filename.replace('.ply', '.binvox'))

            voxelization(input_file, output_file, voxel_size)

if __name__ == "__main__":
    main()