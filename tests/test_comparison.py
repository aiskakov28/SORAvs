"""
Unit tests for video comparison pipeline.
"""

import unittest
import os
import numpy as np
from src.video_comparison import VideoComparisonPipeline
from src.feature_extraction import FeatureExtractor, extract_frames
from src.bias_analysis import BiasAnalyzer

class TestVideoComparisonPipeline(unittest.TestCase):

    def setUp(self):
        self.original_dir = "data/original_videos"
        self.sora_dir = "data/sora_videos"
        self.output_dir = "data/test_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.pipeline = VideoComparisonPipeline(
            original_dir=self.original_dir,
            sora_dir=self.sora_dir,
            output_dir=self.output_dir
        )

    def test_initialization(self):
        self.assertEqual(self.pipeline.original_dir, self.original_dir)
        self.assertEqual(self.pipeline.sora_dir, self.sora_dir)
        self.assertEqual(self.pipeline.output_dir, self.output_dir)

        self.assertIsInstance(self.pipeline.feature_extractor, FeatureExtractor)
        self.assertIsInstance(self.pipeline.bias_analyzer, BiasAnalyzer)

    def test_reset_results(self):
        self.pipeline.reset_results()
        self.assertEqual(len(self.pipeline.results['video_pairs']), 0)
        self.assertEqual(len(self.pipeline.results['structural_similarity']), 0)
        self.assertEqual(len(self.pipeline.results['feature_similarity']), 0)
        self.assertEqual(len(self.pipeline.results['motion_similarity']), 0)
        self.assertEqual(len(self.pipeline.results['color_distribution']), 0)
        self.assertEqual(len(self.pipeline.results['bias_metrics']), 0)

    def test_update_results(self):
        self.pipeline.reset_results()
        test_result = {
            'original_video': 'test1.mp4',
            'sora_video': 'test1_sora.mp4',
            'ssim': 0.8,
            'feature_distance': 50.0,
            'motion_difference': 2.0,
            'color_difference': 0.3
        }

        self.pipeline.update_results(test_result)
        self.assertEqual(len(self.pipeline.results['video_pairs']), 1)
        self.assertEqual(self.pipeline.results['video_pairs'][0], 'test1.mp4')
        self.assertEqual(self.pipeline.results['structural_similarity'][0], 0.8)
        self.assertEqual(self.pipeline.results['feature_similarity'][0], 50.0)
        self.assertEqual(self.pipeline.results['motion_similarity'][0], 2.0)
        self.assertEqual(self.pipeline.results['color_distribution'][0], 0.3)

    @unittest.skipIf(not os.path.exists('data/original_videos/Seef_Heritage.mp4'),
                     "Test video file not found")
    def test_preprocess_video(self):
        video_path = os.path.join(self.original_dir, 'Seef_Heritage.mp4')
        if os.path.exists(video_path):
            frames = self.pipeline.preprocess_video(video_path)
            self.assertGreater(len(frames), 0)
            self.assertEqual(frames[0].shape[2], 3)  # RGB channels

    @unittest.skipIf(not os.path.exists('data/original_videos/Seef_Heritage.mp4') or
                     not os.path.exists('data/sora_videos/Sora example.mp4'),
                     "Test video files not found")
    def test_compare_videos(self):
        original_video = os.path.join(self.original_dir, 'Seef_Heritage.mp4')
        sora_video = os.path.join(self.sora_dir, 'Sora example.mp4')
        if os.path.exists(original_video) and os.path.exists(sora_video):
            result = self.pipeline.compare_videos(original_video, sora_video)

            self.assertIsNotNone(result)
            self.assertEqual(result['original_video'], original_video)
            self.assertEqual(result['sora_video'], sora_video)
            self.assertGreaterEqual(result['ssim'], 0.0)
            self.assertLessEqual(result['ssim'], 1.0)
            self.assertGreaterEqual(result['feature_distance'], 0.0)


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = FeatureExtractor()
    def test_initialization(self):
        self.assertEqual(self.feature_extractor.model_name, 'VGG16')
        self.assertEqual(self.feature_extractor.layer_name, 'block5_conv3')
    def test_extract_features_shape(self):
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        test_image[50:150, 50:150, 0] = 255  # Red square
        features = self.feature_extractor.extract_features(test_image)
        self.assertEqual(features.shape[0], 512 * 14 * 14)

if __name__ == '__main__':
    unittest.main()
