import torch
from torch import nn
from torchvision import models


class PoseNetModified(nn.Module):
    """
    PoseNet implementation with geometric loss function and VGG feature extraction as base network.
    References:
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf
        https://arxiv.org/pdf/1704.00390.pdf
    """

    def __init__(self, pretrained=False):
        super().__init__()
        # Isolate the feature extraction layers only (PyTorch provides this as `features` module in vgg model)
        self.vgg_features = models.vgg16_bn(pretrained=pretrained).features
        self.linear_regression_layer_pos = nn.Linear(in_features=25088, out_features=3)
        self.linear_regression_layer_rot = nn.Linear(in_features=25088, out_features=4)

    def forward(self, input_image):
        # TODO: Perform image transformation

        # Extract features
        out = self.vgg_features(input_image)
        # Regress to position(3D) + quaternion(4D) vector
        out_pos = self.linear_regression_layer_pos(out)
        out_rot = self.linear_regression_layer_rot(out)
        # Normalize the quaternion vector
        normalized_rot = torch.div(out_rot, torch.norm(out_rot))

        return torch.cat([out_pos, normalized_rot], 1)


conjugation_tensor = torch.from_numpy(np.array([1, -1, -1, -1]))


# Performs reprojection given a camera rotation and position
def reprojection(camera_rotation, camera_position, point_3d):
    """
    Reprojects the 3D point into pixel space.
    We ignore the instrinsic scale parameters since it is assumed to be the same for all images (same camera).
    :param camera_rotation: 3D quaternion (w p q r)
    :param camera_position: 3D camera position as (x, y, z, 1)
    :param point_3d: 3D point to be projected (p_x, p_y, p_z, 1)
    :return:
    """
    # [R' | -R'C] * [x y z 1]' = [R' * [x y z]' - R'C . [x y z]'  1]
    camera_rotation_conjugate = camera_rotation * conjugation_tensor
    pixel_coords = camera_rotation * point_3d * camera_rotation_conjugate - camera_rotation * camera_position * camera_rotation_conjugate * point_3d
    pixel_coords /= pixel_coords[2]
    return pixel_coords[:2]


def reprojection_loss(X, y):
    return nn.MSELoss(reprojection(X[:, :4], X[:, 4:7], all_points), y)
