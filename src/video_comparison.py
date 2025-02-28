"""Enhanced video comparison module for analyzing Sora-generated AI videos vs original videos."""

import os
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
import matplotlib.pyplot as plt

from src.feature_extraction import (
    FeatureExtractor, compute_color_histogram, compute_motion_vectors,
    extract_frames, detect_scenes, estimate_depth
)
from src.bias_analysis import BiasAnalyzer
from src.visualization import VisualizationGenerator

class VideoComparisonPipeline:
    def __init__(self, original_dir, sora_dir, output_dir, config=None):
        self.original_dir = original_dir
        self.sora_dir = sora_dir
        self.output_dir = output_dir
        self.config = config or {}
        self.custom_pairs = None

        os.makedirs(output_dir, exist_ok=True)

        model_name = self.config.get('feature_model', 'VGG16')
        self.feature_extractor = FeatureExtractor(
            model_name=model_name,
            use_lpips=self.config.get('use_lpips', True)
        )

        self.bias_analyzer = BiasAnalyzer()

        self.viz_generator = VisualizationGenerator(
            output_dir=output_dir,
            dpi=self.config.get('plot_dpi', 300),
            color_palette=self.config.get('color_palette', 'coolwarm')
        )

        self.reset_results()

    def reset_results(self):
        self.results = {
            'video_pairs': [],
            'structural_similarity': [],
            'feature_similarity': [],
            'motion_similarity': [],
            'color_distribution': [],
            'lpips_score': [],
            'object_accuracy': [],
            'perceived_quality': [],
            'bias_metrics': {}
        }

    def preprocess_video(self, video_path):
        max_frames = self.config.get('max_frames', 100)
        sample_rate = self.config.get('frame_sample_rate', 5)
        frames = extract_frames(video_path, max_frames=max_frames, sample_interval=sample_rate)
        print(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")
        return frames

    def compare_frames(self, original_frame, sora_frame):
        if original_frame.shape != sora_frame.shape:
            sora_frame = cv2.resize(sora_frame, (original_frame.shape[1], original_frame.shape[0]))

        orig_gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
        sora_gray = cv2.cvtColor(sora_frame, cv2.COLOR_RGB2GRAY)

        ssim_score, _ = structural_similarity(orig_gray, sora_gray, full=True)

        orig_features = self.feature_extractor.extract_features(original_frame)
        sora_features = self.feature_extractor.extract_features(sora_frame)

        feature_distance = np.linalg.norm(orig_features - sora_features)

        orig_hist = compute_color_histogram(original_frame)
        sora_hist = compute_color_histogram(sora_frame)
        color_distance = np.linalg.norm(orig_hist - sora_hist)

        lpips_distance = self.feature_extractor.compute_lpips(original_frame, sora_frame)

        original_depth = estimate_depth(original_frame)
        sora_depth = estimate_depth(sora_frame)

        depth_similarity = 0.0
        if original_depth is not None and sora_depth is not None:
            if original_depth.shape != sora_depth.shape:
                sora_depth = cv2.resize(sora_depth, (original_depth.shape[1], original_depth.shape[0]))

            depth_correlation = np.corrcoef(original_depth.flatten(), sora_depth.flatten())[0, 1]
            depth_similarity = max(0, depth_correlation)

        original_objects = []
        sora_objects = []
        objects_accuracy = 0.0

        if self.feature_extractor.object_detector is not None:
            try:
                original_objects = self.feature_extractor.detect_objects(original_frame)
                sora_objects = self.feature_extractor.detect_objects(sora_frame)
                objects_accuracy = self._calculate_object_accuracy(original_objects, sora_objects)
            except Exception as e:
                print(f"Object detection skipped: {str(e)}")

        return {
            'ssim': ssim_score,
            'feature_distance': feature_distance,
            'color_distance': color_distance,
            'lpips': lpips_distance,
            'original_objects': original_objects,
            'sora_objects': sora_objects,
            'objects_accuracy': objects_accuracy,
            'original_depth': original_depth,
            'sora_depth': sora_depth,
            'depth_similarity': depth_similarity
        }

    def _calculate_object_accuracy(self, original_objects, sora_objects):
        if not original_objects:
            return 1.0 if not sora_objects else 0.0

        original_classes = [obj['class'] for obj in original_objects]
        sora_classes = [obj['class'] for obj in sora_objects]

        matches = 0
        for cls in original_classes:
            if cls in sora_classes:
                matches += 1
                sora_classes.remove(cls)

        recall = matches / len(original_classes) if original_classes else 0.0
        precision = matches / len(sora_objects) if sora_objects else 0.0

        if recall + precision > 0:
            return 2 * recall * precision / (recall + precision)
        else:
            return 0.0

    def compare_videos(self, original_video, sora_video):
        print(f"Comparing videos: {os.path.basename(original_video)} and {os.path.basename(sora_video)}")

        original_frames = self.preprocess_video(original_video)
        sora_frames = self.preprocess_video(sora_video)

        if not original_frames or not sora_frames:
            print(f"Warning: No frames extracted from {original_video} or {sora_video}")
            return None

        min_frames = min(len(original_frames), len(sora_frames))
        original_frames = original_frames[:min_frames]
        sora_frames = sora_frames[:min_frames]

        original_scenes = detect_scenes(original_video)
        sora_scenes = detect_scenes(sora_video)

        ssim_scores = []
        feature_distances = []
        color_distances = []
        lpips_scores = []
        object_accuracy_scores = []
        depth_similarity_scores = []
        original_objects_all = []
        sora_objects_all = []

        for i in tqdm(range(min_frames), desc="Comparing frames"):
            metrics = self.compare_frames(original_frames[i], sora_frames[i])
            ssim_scores.append(metrics['ssim'])
            feature_distances.append(metrics['feature_distance'])
            color_distances.append(metrics['color_distance'])
            lpips_scores.append(metrics['lpips'])
            object_accuracy_scores.append(metrics['objects_accuracy'])

            if metrics.get('depth_similarity') is not None:
                depth_similarity_scores.append(metrics['depth_similarity'])

            original_objects_all.append(metrics['original_objects'])
            sora_objects_all.append(metrics['sora_objects'])

            if i % 10 == 0 or i == min_frames - 1:
                self.viz_generator.create_object_detection_visualization(
                    original_frames[i], sora_frames[i],
                    metrics['original_objects'], metrics['sora_objects'],
                    os.path.splitext(os.path.basename(original_video))[0], i
                )

                if metrics['original_depth'] is not None and metrics['sora_depth'] is not None:
                    self.viz_generator.create_depth_map_comparison(
                        metrics['original_depth'], metrics['sora_depth'],
                        os.path.splitext(os.path.basename(original_video))[0], i
                    )

        orig_motion = compute_motion_vectors(original_frames)
        sora_motion = compute_motion_vectors(sora_frames)

        if len(orig_motion) > 0 and len(sora_motion) > 0:
            motion_diff = np.mean(np.abs(orig_motion - sora_motion[:len(orig_motion)]))
        else:
            motion_diff = np.nan

        avg_ssim = np.mean(ssim_scores)
        avg_feature_distance = np.mean(feature_distances)
        avg_color_distance = np.mean(color_distances)
        avg_lpips = np.mean(lpips_scores)
        avg_object_accuracy = np.mean(object_accuracy_scores)

        avg_depth_similarity = np.mean(depth_similarity_scores) if depth_similarity_scores else 0.0

        scene_analysis = self.bias_analyzer.analyze_scene_transitions(
            original_scenes, sora_scenes
        )

        semantic_analysis = self.bias_analyzer.analyze_semantic_content(
            original_objects_all, sora_objects_all
        )

        if original_scenes and sora_scenes:
            self.viz_generator.create_scene_transition_comparison(
                original_scenes, sora_scenes,
                int(max([end for _, end in original_scenes]) if original_scenes else 0),
                int(max([end for _, end in sora_scenes]) if sora_scenes else 0),
                os.path.splitext(os.path.basename(original_video))[0]
            )

        output_prefix = f"{os.path.splitext(os.path.basename(original_video))[0]}"

        self.viz_generator.create_frame_by_frame_comparison(
            ssim_scores, feature_distances, lpips_scores, output_prefix
        )

        quality_metrics = {
            'ssim': avg_ssim,
            'lpips': avg_lpips,
            'feature_distance': avg_feature_distance,
            'motion_difference': motion_diff if not np.isnan(motion_diff) else 0.0,
            'color_difference': avg_color_distance,
            'objects_accuracy': avg_object_accuracy
        }

        analysis_result = self.bias_analyzer.analyze_single_video_pair(quality_metrics)

        self.viz_generator.create_single_video_comparison(
            original_frames, sora_frames, analysis_result, output_prefix
        )

        return {
            'original_video': original_video,
            'sora_video': sora_video,
            'ssim': avg_ssim,
            'feature_distance': avg_feature_distance,
            'motion_difference': motion_diff,
            'color_difference': avg_color_distance,
            'lpips': avg_lpips,
            'objects_accuracy': avg_object_accuracy,
            'depth_similarity': avg_depth_similarity,
            'perceived_quality': analysis_result['similarity_score'],
            'frame_metrics': {
                'ssim_scores': ssim_scores,
                'feature_distances': feature_distances,
                'color_distances': color_distances,
                'lpips_scores': lpips_scores,
                'object_accuracy_scores': object_accuracy_scores
            },
            'scene_analysis': scene_analysis,
            'semantic_analysis': semantic_analysis,
            'analysis': analysis_result
        }

    def run_pipeline(self, bias_categories=None):
        self.reset_results()

        if bias_categories is None:
            bias_categories = self.config.get('bias_categories', {
                'gender': ['male', 'female', 'nonbinary'],
                'ethnicity': ['asian', 'black', 'hispanic', 'white'],
                'age': ['child', 'young', 'adult', 'elderly'],
                'setting': ['urban', 'rural', 'indoor', 'outdoor']
            })

        comparison_results = []

        if self.custom_pairs:
            for pair in self.custom_pairs:
                result = self.compare_videos(pair['original'], pair['sora'])
                if result:
                    comparison_results.append(result)
                    self.update_results(result)
        else:
            original_videos = [os.path.join(self.original_dir, f) for f in os.listdir(self.original_dir)
                              if f.endswith(('.mp4', '.avi', '.mov'))]

            for original_video in original_videos:
                base_name = os.path.basename(original_video)
                sora_video = os.path.join(self.sora_dir, base_name)

                if os.path.exists(sora_video):
                    result = self.compare_videos(original_video, sora_video)
                    if result:
                        comparison_results.append(result)
                        self.update_results(result)
                else:
                    print(f"Warning: No matching Sora video for {original_video}")

        bias_metrics = self.bias_analyzer.analyze_bias(comparison_results, bias_categories)
        self.results['bias_metrics'] = bias_metrics

        bias_df = self.bias_analyzer.compute_bias_scores(bias_metrics)

        problematic_bias = self.bias_analyzer.detect_problematic_bias(bias_df)
        self.results['problematic_bias'] = problematic_bias

        self.save_results(comparison_results, bias_df)

        self.generate_visualizations(bias_metrics)

        return self.results

    def update_results(self, result):
        self.results['video_pairs'].append(os.path.basename(result['original_video']))
        self.results['structural_similarity'].append(result['ssim'])
        self.results['feature_similarity'].append(result['feature_distance'])
        self.results['motion_similarity'].append(result['motion_difference'])
        self.results['color_distribution'].append(result['color_difference'])

        if 'lpips' in result:
            self.results['lpips_score'].append(result['lpips'])
        if 'objects_accuracy' in result:
            self.results['object_accuracy'].append(result['objects_accuracy'])
        if 'perceived_quality' in result:
            self.results['perceived_quality'].append(result['perceived_quality'])

    def save_results(self, comparison_results, bias_df):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(os.path.join(self.output_dir, f'comparison_results_{timestamp}.json'), 'w') as f:
            serializable_results = []
            for result in comparison_results:
                serializable_result = {}
                for key, value in result.items():
                    if key == 'frame_metrics':
                        serializable_result[key] = {}
                        for metric_name, metric_values in value.items():
                            if hasattr(metric_values, 'tolist'):
                                serializable_result[key][metric_name] = metric_values.tolist()
                            else:
                                serializable_result[key][metric_name] = [float(v) if hasattr(v, 'item') else v for v in metric_values]
                    elif isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    elif hasattr(value, 'item'):
                        serializable_result[key] = value.item()
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)

            for result in serializable_results:
                if 'analysis' in result:
                    for key, value in result['analysis'].items():
                        if isinstance(value, dict):
                            for subkey, subvalue in list(value.items()):
                                if hasattr(subvalue, 'item'):
                                    result['analysis'][key][subkey] = subvalue.item()

            serializable_output = {
                'metadata': {
                    'timestamp': timestamp,
                    'num_videos': len(comparison_results)
                },
                'results': serializable_results,
                'bias_metrics': self._convert_to_serializable(self.results['bias_metrics']),
                'problematic_bias': self._convert_to_serializable(self.results.get('problematic_bias', {}))
            }

            json.dump(serializable_output, f, indent=2)

        df = pd.DataFrame({
            'video_pair': self.results['video_pairs'],
            'structural_similarity': self.results['structural_similarity'],
            'feature_similarity': self.results['feature_similarity'],
            'motion_similarity': self.results['motion_similarity'],
            'color_distribution': self.results['color_distribution']
        })

        if self.results['lpips_score']:
            df['lpips_score'] = self.results['lpips_score']
        if self.results['object_accuracy']:
            df['object_accuracy'] = self.results['object_accuracy']
        if self.results['perceived_quality']:
            df['perceived_quality'] = self.results['perceived_quality']

        df.to_csv(os.path.join(self.output_dir, f'comparison_metrics_{timestamp}.csv'), index=False)

        if not bias_df.empty:
            bias_df.to_csv(os.path.join(self.output_dir, f'bias_metrics_{timestamp}.csv'), index=False)

    def _convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def generate_visualizations(self, bias_metrics):
        df = pd.DataFrame({
            'video_pair': self.results['video_pairs'],
            'structural_similarity': self.results['structural_similarity'],
            'feature_similarity': self.results['feature_similarity'],
            'motion_similarity': self.results['motion_similarity'],
            'color_distribution': self.results['color_distribution']
        })

        if self.results['lpips_score']:
            df['lpips_score'] = self.results['lpips_score']
        if self.results['object_accuracy']:
            df['object_accuracy'] = self.results['object_accuracy']
        if self.results['perceived_quality']:
            df['perceived_quality'] = self.results['perceived_quality']

        self.viz_generator.create_metric_distributions(df)
        self.viz_generator.create_correlation_matrix(df)
        self.viz_generator.create_bias_category_plots(bias_metrics)
