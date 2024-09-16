import os, argparse
import glob
import argparse
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from PIL import Image
import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()

# NuScene only extracts bounding boxes through a detection model
# since the bounding boxes weren't provided by the dataset. But
# rest of the info are and are easily accessible via indexing 
# extracted from the json file and used in NuScenes objects.
# On the otherhand, Waymo Dataset requires us to extract key
# information for each frame from the binary data and go
# through a manual categorization process, in addition to identifying
# bounding boxes, so more work.

def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def save_scene_label(frame_object, frame_folder):
    '''
    Extract scene label information and save it in frame folder.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Contains frame information to be extracted
        frame_folder : str
            Path to the frame object's extraction save folder
    Returns:
        scene_label : np.array
            1D array that contains frame time of day and weather information.
    '''
    # Extract and construct scene labels
    frame_time_of_day = frame_object.context.stats.time_of_day
    frame_weather = frame_object.context.stats.weather
    scene_label = np.array([frame_time_of_day, frame_weather])
    # Save and Return
    scene_label_filename = "scene_label.npy"
    scene_label_save_filepath = os.path.join(frame_folder, scene_label_filename)
    np.save(scene_label_save_filepath, scene_label)
    return scene_label

def save_frame_pose(frame_object, frame_folder):
    '''
    Extract frame pose information and save it in frame folder.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Contains frame information to be extracted
        frame_folder : str
            Path to the frame object's extraction save folder
    Returns:
        frame_pose : np.array
            4x4 array that contains frame pose information
    '''
    # Extract and construct frame pose
    frame_pose = np.reshape(np.array(frame_object.pose.transform), [4, 4])
    # Save and Return
    frame_pose_filename = 'vehicle_to_global_pose.npy'
    frame_pose_save_filepath = os.path.join(frame_folder, frame_pose_filename)
    np.save(frame_pose_save_filepath, frame_pose)
    return frame_pose

def save_camera_intrinsics_and_extrinsics(frame_object, frame_folder):
    '''
    Extract frame camera intrinsics and extrinsics information and save them in frame folder.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Contains frame information to be extracted
        frame_folder : str
            Path to the frame object's extraction save folder
    Returns:
        camera_front_intrinsics : np.array
            3x3 array that contains frame front camera intrinsics information
        camera_front_extrinsics : np.array
            4x4 array that contains frame front camera extrinsics information
    '''
    # Extract and construct frame front camera's intrinsics and extrinsics matrices
    camera_calibrations = frame_object.context.camera_calibrations
    for cam_info in camera_calibrations:
        if cam_info.name == open_dataset.CameraName.FRONT:
            camera_front_intrinsics = np.reshape(np.array(cam_info.intrinsic), [3, 3])
            camera_front_extrinsics = np.reshape(np.array(cam_info.extrinsic.transform), [4, 4])
            break
    # Save and Return (NOTE: no error checking, because they should exist else bad dataset quality)
    intrinsics_filename = "front_cam_intrinsics.npy"
    front_intrinsics_save_filepath = os.path.join(frame_folder, intrinsics_filename)
    np.save(front_intrinsics_save_filepath, camera_front_intrinsics)
    extrinsics_filename = "camera_to_vehicle.npy"
    front_extrinsics_save_filepath = os.path.join(frame_folder, extrinsics_filename)
    np.save(front_extrinsics_save_filepath, camera_front_extrinsics)
    return camera_front_intrinsics, camera_front_extrinsics

def save_front_image(frame_object, frame_folder):
    '''
    Extract frame front camera image and save it in frame folder.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Contains frame information to be extracted
        frame_folder : str
            Path to the frame object's extraction save folder
    Returns:
        image : Image array
            The frame front camera image in jpeg format
        image_shape : tuple
            The shape tuple of the jpeg image
    '''
    # Extract and construct frame front camera's image
    for image_token in frame_object.images:
        if image_token.name == open_dataset.CameraName.FRONT:
            image = tf.image.decode_jpeg(image_token.image).numpy()
            image_shape = image.shape
            image = Image.fromarray(np.uint8(image))
    # Save and Return (NOTE: no error checking, because they should exist else bad dataset quality)
    image_filename = 'front_camera.jpeg'
    image_save_filepath = os.path.join(frame_folder, image_filename)
    image.save(image_save_filepath)
    return image, image_shape
    
def save_sparse_depth(z, frame_folder, multiplier=256.0):
    '''
    Saves a sparse depth map to a 16-bit PNG file in frame folder

    Arg(s):
        z : numpy[float32]
            depth map
        frame_folder : str
            path to frame object's extraction save folder to store depth map
        multiplier : float
            multiplier for encoding float as unsigned integer
    '''
    # Save
    sparse_depth_filename = "sparse_depth.png"
    sparse_depth_save_filepath = os.path.join(frame_folder, sparse_depth_filename)
    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(sparse_depth_save_filepath)

def save_bbox(bbox, frame_folder):
    '''
    Saves bounding box numpy tensor in frame folder
    '''
    # Save
    bbox_filename = 'bboxes.npy'
    bbox_save_path = os.path.join(frame_folder, bbox_filename)
    np.save(bbox_save_path, bbox)

def save_image_metadata(frame, save_dir):
    '''
    Saves image metadata in save_dir folder
    Excuse the abrupt difference in convention; this function was copied on
    later on as a quick patch since it wasn't in the old code. So didn't bother
    to fix. Very small function though, should make great sense still.
    Just a simpler extract and save.
    '''
    camera_images = frame.images
    for camera_image in camera_images:
        if camera_image.name == open_dataset.CameraName.FRONT:
            velocity_object = camera_image.velocity
            velocity_array = [velocity_object.v_x, velocity_object.v_y, velocity_object.v_z, velocity_object.w_x, velocity_object.w_y, velocity_object.w_z]
            metadata_array = np.array(velocity_array)
            metadata_array = np.append(metadata_array,
                                    [camera_image.pose_timestamp,
                                    camera_image.shutter,
                                    camera_image.camera_trigger_time,
                                    camera_image.camera_readout_done_time])
            metadata_save_path = os.path.join(save_dir,'camera_metadata.npy')
            np.save(metadata_save_path,metadata_array)

def parse_range_image_and_camera_proj(frame_object, main_laser_names):
    """
    Parse range images and camera projections given a frame.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Open dataset frame proto, contains frame information
        main_laser_names : list[string/enum]
            A list of LIDAR laser names/perspectives that we are interested in
    Returns:
        range_images: 
            A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: 
            A dict of {laser_name, [camera_projection_from_first_return, 
            camera_projection_from_second_return]}.
        range_image_top_pose: 
            range image pixel pose for top lidar (share same reference as the one in range_images).
    """
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None

    # For each laser persepctive in frame's lasers
    for laser in frame_object.lasers:
        # Only care about lasers of interest
        if laser.name in main_laser_names:
            # If there's range image detected by this laser
            if len(laser.ri_return1.range_image_compressed) > 0:
                # Retrieve range image tensor and store it as the value of the corresponding laser key.
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                range_images[laser.name] = [ri]  # one-to-one

                # Special case to be aware of: TOP Laser
                if laser.name == open_dataset.LaserName.TOP:
                    # NOTE: Need to make a deepcopy again, otherwise data is altered later.
                    range_image_top_pose_str_tensor = tf.io.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                    range_image_top_pose = open_dataset.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytearray(range_image_top_pose_str_tensor.numpy()))

                # Retrieve camera projection tensor and store it as the value of the corresponding laser key.
                camera_projection_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.camera_projection_compressed, 'ZLIB')
                cp = open_dataset.MatrixInt32()
                cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                camera_projections[laser.name] = [cp]

    return range_images, camera_projections, range_image_top_pose

# no code/logic changes
def point_cloud_from_range_image(frame_object,
                                 range_images,
                                 camera_projections,
                                 range_image_top_pose,
                                 main_laser_names,
                                 ri_index=0,
                                 keep_polar_features=False):
    """Convert range images to point cloud.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Open dataset frame proto, contains frame information
        range_images: 
            A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: 
            A dict of {laser_name, [camera_projection_from_first_return,
            camera_projection_from_second_return]}.
        range_image_top_pose: 
            range image pixel pose for top lidar.
        main_laser_names : list[string/enum]
            A list of LIDAR laser names/perspectives that we are interested in
        ri_index : int
            0 for the first return, 1 for the second return.
        keep_polar_features : bool
            If true, keep the features from the polar range image
            (i.e. range, intensity, and elongation) as the first features in the
            output range image.
    Returns:
        points: 
            {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
            (NOTE: Will be {[N, 6]} if keep_polar_features is true.
        cp_points: 
            {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    # NOTE: Calibration is lidar calibration, so share same name as lidar laser since belonging to it.
    calibrations = sorted(frame_object.context.laser_calibrations, key=lambda c: c.name)  # sort by name.
    points = []
    cp_points = []

    # Convert range images from base to polar to cartesian coordinates
    cartesian_range_images = convert_range_image_to_cartesian(
        frame_object, range_images, range_image_top_pose, main_laser_names, ri_index, keep_polar_features)

    # Process point cloud points for each laser calibration
    for c in calibrations:
        # Only care about lidar lasers of interest 
        if c.name in main_laser_names:
            # Grab its laser's range image and set up mask
            range_image = range_images[c.name][ri_index]  # grab first return by default, pretty much
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            # Grab its laser's cartesian range image and valid slices based on validity mask
            range_image_cartesian = cartesian_range_images[c.name]
            points_tensor = tf.gather_nd(range_image_cartesian,
                                        tf.compat.v1.where(range_image_mask))

            # Grab its laser's camera projection and valid slices based on validity mask
            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor,
                                            tf.compat.v1.where(range_image_mask))
            
            # Add the catesian sparse depth point map tensor and camera projection tensor into the collections
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

    return points, cp_points

# no code/logic changes
def convert_range_image_to_cartesian(frame_object,
                                     range_images,
                                     range_image_top_pose,
                                     main_lidar_names,
                                     ri_index=0,
                                     keep_polar_features=False):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Arg(s):
        frame_object : waymo_open_dataset.Frame
            Open dataset frame proto, contains frame information
        range_images: 
            A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_image_top_pose: 
            range image pixel pose for top lidar.
        main_laser_names : list[string/enum]
            A list of LIDAR laser names/perspectives that we are interested in
        ri_index: 
            0 for the first return, 1 for the second return
        keep_polar_features: 
            If true, keep the features from the polar range image (i.e. range, intensity, and elongation) 
            as the first features in the output range image.
    Returns:
        cartesian_range_images
            dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
            will be 3 if keep_polar_features is False (x, y, z) and 6 if
            keep_polar_features is True (range, intensity, elongation, x, y, z).
    """
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame_object.pose.transform), 
                                                                   [4, 4]))  # grab the frame pose

    # Grab the range image top pose tensor
    # Dim: [H, W, 6] -- first are roll, pitch, yaw, last three - translation
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    
    # Grab the rotation and translation matrices from top pose tensor, 
    # then reconstruct the top pose tensor
    # Output Dim: [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    
    # Process each laser calibration
    for c in frame_object.context.laser_calibrations:
        # Only care about lidar lasers of interest 
        if c.name in main_lidar_names:
            # Grab its laser's range image
            range_image = range_images[c.name][ri_index]

            # Grab or construct laser's beam inclinations
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)
            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])  # formatting
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)

            # Set up local pixel pose and frame pose
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)

            # Set up expanded extrinsics
            
            extrinsics_expanded = tf.expand_dims(extrinsic, axis=0)

            # Compute the polar coordinates of range image
            
            range_image_polar = compute_range_image_polar(
                range_image=tf.expand_dims(range_image_tensor[..., 0], axis=0),  #[B,H,W]
                extrinsic=extrinsics_expanded,  #[B,4,4]
                #[B,H]-for each row of range image
                inclination=tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                dtype=tf.float32,
                scope=None)

            # Compute the cartesian coordinates of range image
            range_image_cartesian = compute_range_image_cartesian(
                range_image_polar = range_image_polar,
                extrinsic=extrinsics_expanded,
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local,
                dtype=tf.float32,
                scope=None)
            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

            if keep_polar_features:
                # If we want to keep the polar coordinate features of range, intensity,
                # and elongation, concatenate them to be the initial dimensions of the
                # returned Cartesian range image.
                range_image_cartesian = tf.concat(
                    [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)
            cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images

# no code/logic changes.
def get_rotation_matrix(roll, pitch, yaw, name=None):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Arg(s):
        roll : 
            x-rotation in radians.
        pitch:
            y-rotation in radians. The shape must be the same as roll.
        yaw: 
            z-rotation in radians. The shape must be the same as roll.
        name: 
            the op name.
    Returns:
        A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """
    # Prepare data
    cos_roll = tf.cos(roll)
    sin_roll = tf.sin(roll)
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    cos_pitch = tf.cos(pitch)
    sin_pitch = tf.sin(pitch)

    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    # Construct rotation tensor from data
    r_roll = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        tf.stack([zeros, sin_roll, cos_roll], axis=-1),
    ],
                        axis=-2)
    r_pitch = tf.stack([
        tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                        axis=-2)
    r_yaw = tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                        axis=-2)

    return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))

# no code/logic changes.
def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

    Arg(s):
        rotation: 
            [..., N, N] rotation tensor.
        translation: 
            [..., N] translation tensor. This must have the same type as
            rotation.
    Returns:
        transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
        rotation.
    """
    # [..., N, 1]
    translation_n_1 = translation[..., tf.newaxis]  # Current bug: translation's N != 3, whereas Rotation's is.
    # [..., N, N+1]
    transform = tf.concat([rotation, translation_n_1], axis=-1)
    # [..., N]
    last_row = tf.zeros_like(translation)

    last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
    # [..., N+1, N+1]
    transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
    return transform

# no code/logic changes
def compute_inclination(inclination_range, height, scope=None):
    """Computes uniform inclination range based the given range and height.

    Arg(s):
        inclination_range: 
            [..., 2] tensor. Inner dims are [min inclination, max
            inclination].
        height: 
            an integer indicates height of the range image.
        scope: 
            the name scope.
    Returns:
        inclination: [..., height] tensor. Inclinations computed.
    """
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        (.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) /
        tf.cast(height, dtype=inclination_range.dtype) *
        tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1])
    return inclination

# no code/logic changes
def _combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Arg(s):
        tensor: 
            A tensor of any type.
    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    # Prepare data
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []

    # Filter and group
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

# no code/logic changes
def compute_range_image_polar(range_image,
                              extrinsic,
                              inclination,
                              dtype=tf.float32,
                              scope=None):
    """Computes range image polar coordinates.

    Args:
        range_image: [B, H, W] tensor. Lidar range images.
        extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
        inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
        dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
        scope: the name scope.

    Returns:
        range_image_polar: [B, H, W, 3] polar coordinates.
    """
    # Prepare data
    # pylint: disable=unbalanced-tuple-unpacking
    _, height, width = _combined_static_and_dynamic_shape(range_image)
    range_image_dtype = range_image.dtype
    range_image = tf.cast(range_image, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    inclination = tf.cast(inclination, dtype=dtype)

    # Construct polar coordinates from data
    # [B].
    az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
    # [W].
    ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - .5) / tf.cast(
        width, dtype=dtype)
    # [B, W].
    azimuth = (ratios * 2. - 1.) * np.pi - tf.expand_dims(az_correction, -1)

    # [B, H, W]
    azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
    # [B, H, W]
    inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
    range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image],
                                    axis=-1)
    return tf.cast(range_image_polar, dtype=range_image_dtype)

# no code/logic changes
def compute_range_image_cartesian(range_image_polar,
                                  extrinsic,
                                  pixel_pose=None,
                                  frame_pose=None,
                                  dtype=tf.float32,
                                  scope=None):
    """Computes range image cartesian coordinates from polar ones.

    Args:
        range_image_polar: 
            [B, H, W, 3] float tensor. Lidar range image in polar coordinate in sensor frame.
        extrinsic: 
            [B, 4, 4] float tensor. Lidar extrinsic.
        pixel_pose: 
            [B, H, W, 4, 4] float tensor. If not None, it sets pose for each range image pixel.
        frame_pose: 
            [B, 4, 4] float tensor. This must be set when pixel_pose is set.
            It decides the vehicle frame at which the cartesian points are computed.
        dtype: 
            float type to use internally. This is needed as extrinsic and
            inclination sometimes have higher resolution than range_image.
        scope: 
            the name scope.

    Returns:
        range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    # Setup needed data
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    # Validate the data are from the same graph and make the graph the default.
    with tf.compat.v1.name_scope(
        scope, 'ComputeRangeImageCartesian',
        [range_image_polar, extrinsic, pixel_pose, frame_pose]):
        azimuth, inclination, range_image_range = tf.unstack(
            range_image_polar, axis=-1)

        # Carry out math operations
        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = tf.einsum('bkr,bijr->bijk', rotation,
                                    range_image_points) + translation
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = tf.einsum(
                'bhwij,bhwj->bhwi', pixel_pose_rotation,
                range_image_points) + pixel_pose_translation
            if frame_pose is None:
                raise ValueError('frame_pose must be set when pixel_pose is set.')
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = tf.einsum(
                'bij,bhwj->bhwi', world_to_vehicle_rotation,
                range_image_points) + world_to_vehicle_translation[:, tf.newaxis,
                                                                    tf.newaxis, :]

        # Convert and return
        range_image_points = tf.cast(
            range_image_points, dtype=range_image_polar_dtype)
        return range_image_points

parser = argparse.ArgumentParser()
parser.add_argument('--waymo_tfrecord_dir',     
    type=str, required=True, help='Path to Waymo dataset\'s tfrecord directory')
parser.add_argument('--waymo_extracted_save_dir',   
    type=str, required=True, help='Path to save the extracted/derived dataset to')
args = parser.parse_args()


if __name__ == '__main__':
    data_gt_root_dirpath = "/media/data1/derived/waymo_extracted" 
    gt_search_pattern = os.path.join(data_gt_root_dirpath, '**', "segment-*", "frame_*")
    gt_file_paths = glob.glob(gt_search_pattern)
    segment_frame_dict = {}
    for path in gt_file_paths:
        segment = os.path.basename(os.path.split(path)[0])
        if not segment_frame_dict.get(segment):
            segment_frame_dict[segment] = []
        frame = os.path.basename(path)
        segment_frame_dict[segment].append((frame, path))

    # Set up directory to save the extracted data to.
    extracted_data_save_dirpath = args.waymo_extracted_save_dir
    if not os.path.exists(extracted_data_save_dirpath):
        os.mkdir(extracted_data_save_dirpath)
    
    # Set up paths to the tfrecords binary files
    waymo_tfrecord_dirpath = args.waymo_tfrecord_dir
    splits = ["training", "validation", "testing"]
    tfrecords = {}
    for split in splits:
        tfrecords_filepath = os.path.join(waymo_tfrecord_dirpath, split, '*.tfrecord')
        tfrecords[split] = glob.glob(tfrecords_filepath)

    # TODO: print number of scenes to process in total?
    # If so, make a list to store all data. Count and print instances first.
    # TODO: also incorporate multithreadding like in process_waymo?
    # Only sequential in the origin file too, though.

    for split in splits:
        # Incorporate split type into output folder path
        extracted_data_save_dirpath = os.path.join(args.waymo_extracted_save_dir, split)
        if not os.path.exists(extracted_data_save_dirpath):
            os.mkdir(extracted_data_save_dirpath)

        # Process each tfrecord
        for idx, tfrecord in enumerate(tfrecords[split]):
            print(f"Processing {idx} tfrecord out of {len(tfrecords[split])}")
            dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')

            # Process each frame in record
            for count, data in enumerate(tqdm(dataset)):  # tqdm adds a progress bar to the iterable

                # Convert byte data to frame object
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                segment_index = frame.context.name
                frame_index = str(frame.timestamp_micros)

                # Construct extraction output folders (segment, then frame)
                segment_folder = os.path.join(extracted_data_save_dirpath,'segment-'+segment_index)
                if not os.path.exists(segment_folder):
                    print("Creating " + segment_folder)  # announce
                    os.mkdir(segment_folder)
                frame_folder = os.path.join(segment_folder,'frame_'+frame_index)
                os.makedirs(frame_folder, exist_ok=True)  # no announce

                '''Begin processing the frame to extract sparse depth and bounding box'''
                print('Processing {}'.format(frame_folder))

                # Save some frame information into the frame folder, denoted by function name
                label = save_scene_label(frame, frame_folder)
                _ = save_frame_pose(frame, frame_folder)
                _, _ = save_camera_intrinsics_and_extrinsics(frame, frame_folder)
                _, image_shape = save_front_image(frame, frame_folder)
                save_image_metadata(frame, frame_folder)

                # Process and save PCL (point cloud array) depth information
                main_laser_names = [open_dataset.LaserName.TOP,
                                            open_dataset.LaserName.FRONT,
                                            open_dataset.LaserName.SIDE_LEFT,
                                            open_dataset.LaserName.SIDE_RIGHT]
                
                # Grab data from frame
                range_images, camera_projections, range_image_top_pose = \
                    parse_range_image_and_camera_proj(frame, main_laser_names)
                
                # Construct cartesian point cloud and camera projection points from range image
                # They represent the distance between lidar points and vehicle frame origin.
                points, cp_points = point_cloud_from_range_image(frame,
                                                        range_images,
                                                        camera_projections,
                                                        range_image_top_pose,
                                                        main_laser_names)
                images = sorted(frame.images, key=lambda i:i.name)
                # Turn lists into [N,3] tensor, then normalize and adjust
                points_all = np.concatenate(points, axis = 0)
                cp_points_all = np.concatenate(cp_points, axis=0)
                points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
                cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

                # Construct a boolean mask on projection validity

                mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

                # Refine the tensors with the validity mask
                cp_points_all_tensor = tf.cast(tf.gather_nd(
                cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
                points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

                # Combine both tensors together to finalize raw data projection
                projected_points_all_from_raw_data = tf.concat(
                    [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
                
                # Construct the sparse depth ground truth based on the results of raw
                # data projection, filtered by visible in-frame coordinates only
                main_lidar_image = np.zeros((image_shape[0], image_shape[1]))
                for point in projected_points_all_from_raw_data:
                    x = int(point[0])
                    y = int(point[1])
                    if main_lidar_image[y,x] != 0:  # closer depth overrides
                        main_lidar_image[y,x] = min(main_lidar_image[y,x], point[2])
                    else:
                        main_lidar_image[y,x] = point[2]

                segment = os.path.basename(os.path.split(frame_folder)[0])
                frame_name = os.path.basename(frame_folder)
                for gt_frame in segment_frame_dict[segment]:
                    # Same frame name, compare the complete depths
                    if frame_name == gt_frame[0]:
                        gt_frame_found = True

                        test_frame_path = frame_folder
                        gt_frame_path = gt_frame[1]

                        print(f"Statistics for {segment} {frame_name} where d1 is test and d2 is gt:\n")
                        print(f"Test path: {frame_folder}\n")
                        print(f"Gt path: {gt_frame[1]}\n")

                        # Load and compare the depths
                        d1 = main_lidar_image
                        d2 = load_depth(os.path.join(gt_frame_path, "sparse_depth.png"))
                        d1_validity_mask = np.where(d1>0, 1, 0)
                        d2_validity_mask = np.where(d2>0, 1, 0)
                        # compare sumed val_d1 and val_d2
                        # Update to mean error later
                        print(f"Number of non-zero point for d1: {d1_validity_mask.sum()}, and d2: {d2_validity_mask.sum()}.\n")
                        print("Accounting all points, are they equal?: " + str(np.sum(np.abs(d1.flatten() - d2.flatten()))) + '\n')
                        print("Difference values: " + str(np.unique(d1.flatten() - d2.flatten())) + '\n')

                # Save the extracted sparse depth information
                save_sparse_depth(main_lidar_image, frame_folder)

                # Load and compare the depths
                d2 = load_depth(os.path.join(frame_folder, "sparse_depth.png"))
                d1_validity_mask = np.where(d1>0, 1, 0)
                d2_validity_mask = np.where(d2>0, 1, 0)
                # compare sumed val_d1 and val_d2
                # Update to mean error later
                print(f"Number of non-zero point for d1: {d1_validity_mask.sum()}, and d2: {d2_validity_mask.sum()}.\n")
                print("Accounting all points, are they equal?: " + str(np.sum(np.abs(d1.flatten() - d2.flatten()))) + '\n')
                print("Difference values: " + str(np.unique(d1.flatten() - d2.flatten())) + '\n')

                exit(1)

                # Extract the bounding boxes detected in this frame by the front camera
                # NOTE: there's no need to run a detection model to find bboxes unlike NuScenes
                # since the bboxes are already provided in frame information!
                bboxes = []
                for camera_label_token in frame.camera_labels:
                    if camera_label_token.name == open_dataset.CameraName.FRONT:
                        for camera_label in camera_label_token.labels:
                            # Only interested in capturing dynamic moving object's ids
                            if(camera_label.type in [1,2,4]): # moving object ids - check label.proto
                                bbox = [int(camera_label.box.center_x),  # as provided
                                        int(camera_label.box.center_y),
                                        int(camera_label.box.width),
                                        int(camera_label.box.length)]
                                bboxes.append(bbox)

                # Save the extracted bounding box information
                save_bbox(bboxes, frame_folder)

            print('Finished {} frames in {}'.format(count+1, tfrecord))


