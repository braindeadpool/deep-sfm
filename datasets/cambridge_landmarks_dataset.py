import os
import logging
from torch.utils.data import Dataset
from skimage import io
import numpy as np

logging.basicConfig()
logger = logging.getLogger()

DATASET_BASE_URL = 'https://www.repository.cam.ac.uk/bitstream/handle/1810/'
SCENE_NAME_TO_URL = {'ShopFacade': DATASET_BASE_URL + '251336/ShopFacade.zip?scene=4&isAllowed=y',
                     'KingsCollege': DATASET_BASE_URL + '251342/KingsCollege.zip?scene=4&isAllowed=y',
                     'Street': DATASET_BASE_URL + '251292/Street.zip?scene=5&isAllowed=y',
                     'OldHospital': DATASET_BASE_URL + '251340/OldHospital.zip?scene=4&isAllowed=y',
                     'StMarysChurch': DATASET_BASE_URL + '251294/StMarysChurch.zip?scene=5&isAllowed=y',
                     'GreatCourt': DATASET_BASE_URL + '251291/GreatCourt.zip?scene=4&isAllowed=y'
                     }


def load_scene(base_dir, scene_name, mode='train'):
    scene_samples = []
    scene_dir = os.path.join(base_dir, scene_name)
    if os.path.isdir(scene_dir):
        dataset_mode_file = os.path.join(scene_dir, 'dataset_{}.txt'.format(mode))
        with open(dataset_mode_file, 'r') as f:
            for line in f.readlines():
                split_tokens = line.strip().split(' ')
                if len(split_tokens) != 8:
                    logger.debug('Ignoring line {}'.format(line))
                    continue
                frame_path = split_tokens[0]
                camera_pose = list(map(float, split_tokens[1:]))

                scene_samples.append([os.path.join(scene_dir, frame_path), np.array(camera_pose)])
    else:
        # Scene doesnt exist on disk. TODO: Fetch from URL
        logger.error("Sequence does not exist on disk. Please download it")
    return scene_samples


class CambridgeLandmarksDataset(Dataset):
    """
    Cambridge Landmarks Dataset from http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset
    """

    def __init__(self, dataset_dir=os.path.join(os.curdir, 'CambridgeLandmarksDataset'), scene_name=None, mode='train'):
        self.dataset_dir = dataset_dir
        self.scene_name = scene_name
        self.mode = mode
        self.samples = []
        # Load the dataset from disk/url
        if scene_name is None:
            # Load all scenes
            for s_name in SCENE_NAME_TO_URL:
                scene_samples = load_scene(dataset_dir, s_name, self.mode)
                self.samples += scene_samples
        else:
            self.samples = load_scene(dataset_dir, scene_name, self.mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, camera_pose = self.samples[idx]
        image = io.imread(image_path)
        return {'image': image, 'camera_pose': camera_pose}
