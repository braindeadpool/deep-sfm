import unittest
import os
from datasets.cambridge_landmarks_dataset import CambridgeLandmarksDataset


class TestCambridgeLandmarksDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = os.path.abspath(os.path.join(os.path.curdir, 'datasets', 'CambridgeLandmarksDataset'))
        self.scene_name = 'ShopFacade'

    def test_scene_load(self):
        cld_train = CambridgeLandmarksDataset(self.dataset_dir, self.scene_name, mode='train')
        self.assertEqual(len(cld_train), 231)
        self.assertEqual(cld_train[0]['image'].shape, (1080, 1920, 3))

        cld_test = CambridgeLandmarksDataset(self.dataset_dir, self.scene_name, mode='test')
        self.assertEqual(len(cld_test), 103)
