import os
import numpy as np
import tensorflow as tf
import h5py
from transform.affine import Affine
import cv2


class TransporterNetworkDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_directory=None, batch_size=8, augment=True, shuffle=True, crop_size=64, n_orentation_bins=18):
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.initial_crop_size = np.ceil(np.sqrt(2 * crop_size ** 2))
        self.n_orentation_bins = n_orentation_bins

        self.indices = self.get_indices()
        self.on_epoch_end()

    def get_indices(self):
        # count number of hdf5 files starting with scene in self.dataset_directory
        scene_files = [f for f in os.listdir(self.dataset_directory) if f.startswith('scene') and f.endswith('.hdf5')]
        return np.arange(len(scene_files))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        ret = self.get_data(batch)
        return ret
    
    def get_data(self, batch):
        position_inputs = []
        orientation_inputs = []
        position_targets = []
        orientation_targets = []
        
        for i in batch:
            scene_file = os.path.join(self.dataset_directory, f'scene_{i:04d}.hdf5')
            with h5py.File(scene_file, 'r') as f:
                heightmap = f['heightmap'][()]
                colormap = f['colormap'][()]
                grasp_pose = f['grasp_pose'][()]
                workspace_bounds = f['workspace_bounds'][()]
            
            resolution = heightmap.shape
            if self.augment:
                max_attempts = 100  # Prevent infinite loops
                attempt = 0
                while attempt < max_attempts:
                    heightmap_aug, colormap_aug, grasp_pose_aug = self.augment_data(
                        heightmap, colormap, grasp_pose, workspace_bounds
                    )
                    grasp_pixel = self.project_grasp_pose(grasp_pose_aug, resolution, workspace_bounds)
                    if (15 <= grasp_pixel[0] < resolution[0] - 15 and 
                        15 <= grasp_pixel[1] < resolution[1] - 15):
                        heightmap = heightmap_aug
                        colormap = colormap_aug
                        grasp_pose = grasp_pose_aug
                        break
                    attempt += 1
            
            position_input = np.concatenate([heightmap[..., np.newaxis], colormap], axis=-1)
            grasp_pose_pixel = self.project_grasp_pose(grasp_pose, resolution, workspace_bounds)
            position_target = self.one_hot_encode_image(grasp_pose_pixel, resolution)
            orientation_input = self.crop_and_rotate_input_at_pixel(position_input, grasp_pose_pixel)
            theta = self.get_orientation_bin(grasp_pose)
            orientation_target = self.one_hot_encode(theta, self.n_orentation_bins)

            position_inputs.append(position_input)
            position_targets.append(position_target)
            orientation_inputs.append(orientation_input)
            orientation_targets.append(orientation_target)

        inputs = [np.stack(position_inputs).astype(np.float32), np.stack(orientation_inputs).astype(np.float32)]
        targets = [np.stack(position_targets), np.stack(orientation_targets)]
        return inputs, targets

    def project_grasp_pose(self, grasp_pose, resolution, workspace_bounds):
        """Projects 3D grasp pose to 2D pixel coordinates
        Args:
            grasp_pose: 4x4 transformation matrix
            resolution: tuple of (height, width)
            workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        # Extract position from transformation matrix
        position = grasp_pose[:3, 3]
        
        # Convert from world coordinates to pixel coordinates
        x_pixel = int((position[0] - workspace_bounds[0][0]) / (workspace_bounds[0][1] - workspace_bounds[0][0]) * resolution[0])
        y_pixel = int((position[1] - workspace_bounds[1][0]) / (workspace_bounds[1][1] - workspace_bounds[1][0]) * resolution[1])
        
        # Clip to image boundaries
        x_pixel = np.clip(x_pixel, 0, resolution[0] - 1)
        y_pixel = np.clip(y_pixel, 0, resolution[1] - 1)
        
        return (x_pixel, y_pixel)

    def one_hot_encode_image(self, pixel_coords, resolution):
        """Creates a one-hot encoded image with a single 1 at the grasp position"""
        target = np.zeros(resolution)
        # the coordinates are switched
        target[pixel_coords[1], pixel_coords[0]] = 1
        return target

    def crop_and_rotate_input_at_pixel(self, input_image, pixel_coords):
        """Crops and creates multiple rotated versions of the input centered at pixel"""
        crops = []
        half_size = int(self.initial_crop_size // 2)
        half_size_crop = int(self.crop_size // 2)
        
        # Extract crop around grasp point
        # the coordinates are switched
        y, x = pixel_coords
        pad_width = ((half_size, half_size), (half_size, half_size), (0, 0))
        padded_image = np.pad(input_image, pad_width, mode='constant')
        x_pad, y_pad = x + half_size, y + half_size
        crop = padded_image[
            x_pad-half_size:x_pad+half_size,
            y_pad-half_size:y_pad+half_size
        ]

        # Create rotated versions (0 to 180 degrees for symmetric gripper)
        center = (crop.shape[0] // 2, crop.shape[1] // 2)
        for angle in np.linspace(0, 180, self.n_orentation_bins, endpoint=False):
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                crop, 
                rotation_matrix, 
                (int(self.initial_crop_size), int(self.initial_crop_size)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            rotated = rotated[
                half_size - half_size_crop:half_size + half_size_crop,
                half_size - half_size_crop:half_size + half_size_crop
            ]
            crops.append(rotated)
            
        return np.stack(crops)

    def get_orientation_bin(self, grasp_pose):
        """Converts grasp orientation to discrete bin index
        Args:
            grasp_pose: 4x4 transformation matrix
        """
        # Convert to Affine object
        grasp_affine = Affine.from_matrix(grasp_pose)
        angle_rad = grasp_affine.rpy[2]
        angle_rad_symmetric = (angle_rad + np.pi) % (np.pi)
        
        # Convert to degrees and normalize to [0, 360)
        angle_deg = angle_rad_symmetric * 180 / np.pi
        
        # Convert to bin index
        bin_size = 180 / self.n_orentation_bins
        bin_index = round(angle_deg / bin_size) % self.n_orentation_bins
        return bin_index

    def one_hot_encode(self, index, n_classes):
        """Creates one-hot encoded vector"""
        target = np.zeros(n_classes)
        target[index] = 1
        return target

    def augment_data(self, heightmap, colormap, grasp_pose, workspace_bounds):
        """Augment the input data with random rotation around workspace center"""
        h, w = heightmap.shape
        
        # Calculate workspace center (in world coordinates)
        workspace_center_x = (workspace_bounds[0][1] + workspace_bounds[0][0]) / 2
        workspace_center_y = (workspace_bounds[1][1] + workspace_bounds[1][0]) / 2
        
        # Random rotation (-180 to 180 degrees)
        angle = np.random.uniform(-180, 180)
        angle_rad = np.deg2rad(angle)
        
        # For images, we rotate in the opposite direction (clockwise) to match the world rotation
        image_angle = -angle
        
        # Create rotation matrix for image (around image center)
        image_center = (w // 2, h // 2)
        rot_mat_2d = cv2.getRotationMatrix2D(image_center, image_angle, 1.0)
        
        # Make sure heightmap is float32 and colormap is uint8
        heightmap = heightmap.astype(np.float32)
        colormap = colormap.astype(np.uint8)
        
        # Get min height value as float for border
        
        # Apply rotation to heightmap
        heightmap_aug = cv2.warpAffine(
            heightmap, 
            rot_mat_2d, 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-0.6
        )
        
        # Apply rotation to colormap (black border = [0,0,0])
        colormap_aug = cv2.warpAffine(
            colormap, 
            rot_mat_2d, 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Create 3D rotation matrix around workspace center
        transform_3d = np.eye(4)
        
        # First translate to origin
        transform_3d[0:3, 3] = [-workspace_center_x, -workspace_center_y, 0]
        
        # Create rotation matrix
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        rot_z = np.array([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Translate back
        translate_back = np.eye(4)
        translate_back[0:3, 3] = [workspace_center_x, workspace_center_y, 0]
        
        # Combine transformations: translate -> rotate -> translate back
        transform_3d = translate_back @ rot_z @ transform_3d
        
        # Transform grasp pose
        grasp_pose_aug = transform_3d @ grasp_pose
        
        return heightmap_aug, colormap_aug, grasp_pose_aug

    def visualize_augmentation(self, heightmap, colormap, grasp_pose, 
                             heightmap_aug, colormap_aug, grasp_pose_aug,
                             workspace_bounds):
        """Visualize original and augmented data"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Show original data
        axes[0,0].imshow(heightmap)
        axes[0,0].set_title('Original Heightmap')
        axes[1,0].imshow(colormap)
        axes[1,0].set_title('Original Colormap')
        
        # Show augmented data
        axes[0,1].imshow(heightmap_aug)
        axes[0,1].set_title('Augmented Heightmap')
        axes[1,1].imshow(colormap_aug)
        axes[1,1].set_title('Augmented Colormap')
        
        # Plot grasp positions
        orig_pixel = self.project_grasp_pose(grasp_pose, heightmap.shape, workspace_bounds)
        aug_pixel = self.project_grasp_pose(grasp_pose_aug, heightmap.shape, workspace_bounds)
        
        axes[0,0].plot(orig_pixel[0], orig_pixel[1], 'r+', markersize=10)
        axes[0,1].plot(aug_pixel[0], aug_pixel[1], 'r+', markersize=10)
        
        plt.show()
