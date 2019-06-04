import airsim
import numpy as np
from airsim import utils
import time
from PIL import Image


def construct_bbox_corners(position, bbox):
    corners = []
    for x_range in [-bbox[0], bbox[0]]:
        for y_range in [-bbox[1], bbox[1]]:
            for z_range in [-bbox[2], bbox[2]]:
                corner = position - [x_range, y_range, z_range]
                corners.append(corner)

    return corners


def project_box(client, bbox_corners, WIDTH_HEIGHT, RGB_ARRAY):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
    response = responses[0]
    # get numpy array
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    # reshape array to 3 channel image array (H X W,  3)
    img2d = img1d.reshape(response.height * response.width, 3)
    #img3d = img1d.reshape(response.height, response.width, 3)

    airsim.write_file('test.png', response.image_data_uint8)

    if RGB_ARRAY in img2d:
        camera_info = client.simGetCameraInfo(0)
        projections = []

        for corner in bbox_corners:
            projections.append(utils.project_3d_point_to_screen(corner,
                                                                camera_info.pose.position.to_numpy_array(),
                                                                camera_info.pose.orientation,
                                                                camera_info.proj_mat.matrix,
                                                                WIDTH_HEIGHT)
                               )
        projections = np.array(projections)
        xmin = int(min(projections[:, 0]))
        xmax = int(max(projections[:, 0]))
        ymin = int(min(projections[:, 1]))
        ymax = int(max(projections[:, 1]))
        return [xmin, ymin, xmax, ymax]
    else:
        return None


WIDTH_HEIGHT = [256, 144]
MESH_NAME = "OrangeBall"
RGB_ARRAY = [42, 174, 203]

client = airsim.VehicleClient()
client.confirmConnection()

# Set Segmentation ID to a known corresponding RGB_array
client.simSetSegmentationObjectID(MESH_NAME, 1)
# Calls GetBoundingBox function, part of WorldSimApi, passing the object name
bbox_pose = client.simGetBoundingBox(MESH_NAME)
# GetBoundingBox returns a pose, with the origin as position and bbox size as orientation
object_position = client.simGetObjectPose(MESH_NAME).position
object_bbox = bbox_pose.orientation.to_numpy_array()/100
# Retrieve corners of bounding box in 3D
bbox_corners = construct_bbox_corners(object_position.to_numpy_array(), object_bbox)
# The 3D points need to be projected back onto the camera screen
print(project_box(client, bbox_corners, WIDTH_HEIGHT, RGB_ARRAY))
responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation)])
airsim.write_file('test.png', responses[0].image_data_uint8)
