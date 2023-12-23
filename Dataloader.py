# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from pyntcloud import PyntCloud  # You may need to install this library
# import open3d as o3d
# from Model_baseline import Generator
#
#
# class PointCloudDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.file_list = os.listdir(data_dir)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         # Load the PLY files (input and label)
#         file_name = self.file_list[idx]
#         input_ply_path = os.path.join(self.data_dir, "y2.ply")
#         # label_ply_path = os.path.join(self.data_dir, "y_gt.ply")
#
#         # Load PLY data using PyntCloud (or other PLY loading library)
#         input_point_cloud = PyntCloud.from_file(input_ply_path).xyz
#         # label_point_cloud = PyntCloud.from_file(label_ply_path).xyz
#
#         # Convert to PyTorch tensors
#         input_point_cloud = torch.FloatTensor(input_point_cloud)
#         # label_point_cloud = torch.FloatTensor(label_point_cloud)
#         # sample = {
#         #     'image': input_point_cloud
#         # }
#
#         return input_point_cloud
#
#
# # Example usage:
# data_dir = "/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/all files"
#
# # Create the dataset
# dataset = PointCloudDataset(data_dir)
#
# # Create a DataLoader with batch size and other options
# batch_size = 32  # Adjust this according to your needs
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import chardet

# def detect_encoding(file_path):
#     with open(file_path, 'rb') as file:
#         result = chardet.detect(file.read())
#     return result['encoding']
#
# file_path = "/Users/pratham/Desktop/Dataset 2/T1/y1.ply"
# detected_encoding = detect_encoding(file_path)
#
# print(f"Detected encoding: {detected_encoding}")

def read_ply_file(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as ply_file:
        data_section = False

        for line in ply_file:
            line = line.strip()

            if data_section:
                parts = line.split()
                if parts:
                    if parts[0] == "3":  # Assuming it's a triangle face
                        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    return vertices, faces


def main():
    file_path = "/Users/pratham/Desktop/Dataset 2/T1/y_gt.ply"
    vertices, faces = read_ply_file(file_path)

    print("Vertices:")
    for vertex in vertices:
        print(vertex)

    print("Faces:")
    for face in faces:
        print(face)


if __name__ == "__main__":
    main()

# The 'point_cloud' variable now contains your 3D point cloud data as a NumPy array.

# import numpy as np
# import os
# import glob
#
# # Set the path to your point cloud dataset directory
# dataset_directory = "/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/all files"
#
# # Initialize variables to store statistics
# latent_dim = None
# num_points = 0
# point_dim = None
#
# # Iterate through the dataset files and analyze the point clouds
# for file_path in glob.glob(os.path.join(dataset_directory, "y2.ply")):
#     # You will need a PLY file parser or a custom function to read point cloud data from files
#     # Assuming you have a function load_point_cloud_from_file()
#     point_cloud = PointCloudDataset(data_dir)
#
#     # Update statistics based on the first point cloud
#     if latent_dim is None:
#         latent_dim = point_cloud.shape[1]  # Assuming the entire point cloud is used as latent_dim
#
#     num_points += point_cloud.shape[0]
#
#     if point_dim is None:
#         point_dim = point_cloud.shape[1]
#
# # Print the estimated values
# print(f"Estimated latent_dim: {latent_dim}")
# print(f"Estimated num_points: {num_points}")
# print(f"Estimated point_dim: {point_dim}")


# for batch in dataloader:
#
#     input_point_clouds= batch['image']
#     print(input_point_clouds)
#     #
#     print(input_point_clouds)
#     print(input_point_clouds.shape)
#     #
#     # # Convert PyTorch tensors to NumPy arrays
#     input_point_clouds = input_point_clouds.numpy()
#     # # label_point_clouds = label_point_clouds.numpy()
#     # #
#     # # Create Open3D PointCloud objects
#     input_pcd = o3d.geometry.PointCloud()
#     input_pcd.points = o3d.utility.Vector3dVector(input_point_clouds[0])
#     input_pcd.paint_uniform_color([1, 0, 0])  # Blue for input
#     # #
#     # # label_pcd = o3d.geometry.PointCloud()
#     # # label_pcd.points = o3d.utility.Vector3dVector(label_point_clouds[0])
#     # # label_pcd.paint_uniform_color([0, 1, 0])  # Red for label
#     # #
#     # # # Visualize the point clouds using Open3D
#     o3d.visualization.draw_geometries([input_pcd])










