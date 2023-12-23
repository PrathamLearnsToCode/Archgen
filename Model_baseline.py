import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, num_points, point_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.point_dim = point_dim

        # Define the generator architecture
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_points * point_dim)

    def forward(self, z):
        # Generator forward pass
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output is in the range [-1, 1]

        # Reshape the output to match the desired point cloud format
        x = x.view(-1, self.num_points, self.point_dim)
        return x


# Example usage:
latent_dim = 100
num_points = 1024  # Number of points in the point cloud
point_dim = 3  # Dimensionality of each point (x, y, z)
batch_size = 32

# Create an instance of the generator
generator = Generator(latent_dim, num_points, point_dim)

# Generate synthetic point clouds
z = torch.randn(batch_size, latent_dim)  # Random input noise
generated_point_clouds = generator(z)

# The generated_point_clouds tensor now contains synthetic point clouds.
# You can use these generated point clouds for further processing or visualization.


