"""
Main script for enhanced Sora AI Video Bias Analysis Pipeline
Date: February 28, 2025
"""

import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from src.video_comparison import VideoComparisonPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='Enhanced Sora AI Video Bias Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--original-dir', type=str,
                        help='Directory containing original videos')
    parser.add_argument('--sora-dir', type=str,
                        help='Directory containing Sora-generated videos')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save results')
    parser.add_argument('--pair', nargs=2, metavar=('ORIGINAL', 'SORA'),
                        help='Manually specify a pair of videos to compare')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum number of frames to process')
    parser.add_argument('--use-lpips', action='store_true',
                        help='Enable LPIPS perceptual similarity metric')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for processing if available')
    parser.add_argument('--disable-object-detection', action='store_true',
                        help='Disable object detection (may improve stability)')
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file: {str(e)}")
        return {}

def setup_environment(use_gpu):
    if use_gpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        return True
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but not available. Using CPU instead.")
        else:
            print("Using CPU for processing.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return False

def check_dependencies():
    print(f"OpenCV version: {cv2.__version__}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    mobilenet_prototxt = 'models/MobileNetSSD_deploy.prototxt'
    mobilenet_model = 'models/MobileNetSSD_deploy.caffemodel'

    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists(mobilenet_prototxt) or not os.path.exists(mobilenet_model):
        print("Downloading MobileNet SSD model files...")
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
                mobilenet_prototxt
            )
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc",
                mobilenet_model
            )
            print("Model files downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model files: {str(e)}")
            print("Object detection will be disabled.")

def print_analysis_summary(results, verbose=True):
    print("\n===== ANALYSIS SUMMARY =====")

    print(f"Number of videos processed: {len(results['video_pairs'])}")

    if results['structural_similarity']:
        print("\nOverall Video Similarity Metrics (Averages):")
        print(f"  Structural Similarity (SSIM): {np.mean(results['structural_similarity']):.3f}")
        print(f"  Feature Distance: {np.mean(results['feature_similarity']):.3f}")
        print(f"  Motion Difference: {np.mean([m for m in results['motion_similarity'] if not np.isnan(m)]):.3f}")
        print(f"  Color Difference: {np.mean(results['color_distribution']):.3f}")

        if 'lpips_score' in results and results['lpips_score']:
            print(f"  Perceptual Similarity (LPIPS): {np.mean(results['lpips_score']):.3f}")

        if 'object_accuracy' in results and results['object_accuracy']:
            print(f"  Object Recognition Accuracy: {np.mean(results['object_accuracy']):.3f}")

        if 'perceived_quality' in results and results['perceived_quality']:
            print(f"  Overall Perceived Quality: {np.mean(results['perceived_quality']):.3f}")

    print("\nBias Analysis Results:")
    problematic_bias = results.get('problematic_bias', {})

    if problematic_bias:
        print("  Potential bias detected in these categories:")
        for category, issues in problematic_bias.items():
            for issue in issues:
                print(f"  - {category.capitalize()}: Favors '{issue['best_value']}' over '{issue['worst_value']}'")
                print(f"    Score difference: {issue['score_difference']:.3f}, Severity: {issue['severity']}")
    else:
        print("  No significant bias detected in the analyzed categories.")

    if verbose:
        print("\nDetailed Category Metrics:")
        for category, values in results['bias_metrics'].items():
            if values:
                print(f"  {category.capitalize()}:")
                for value, metrics in sorted(values.items(),
                                             key=lambda x: x[1].get('avg_perceived_quality', 0),
                                             reverse=True):
                    print(f"    {value}: Quality={metrics.get('avg_perceived_quality', 0):.3f}, "
                          f"SSIM={metrics['avg_ssim']:.3f}")

    print("\nResults have been saved to the output directory.")


def main():
    args = parse_arguments()
    check_dependencies()
    config = load_config(args.config)
    if not config:
        config = {
            'original_dir': 'data/original_videos',
            'sora_dir': 'data/sora_videos',
            'output_dir': 'data/output',
            'max_frames': 100,
            'frame_sample_rate': 5,
            'use_lpips': True,
            'use_gpu': False,
            'use_object_detection': True,
            'plot_dpi': 300,
            'color_palette': 'coolwarm'
        }

    use_gpu = args.gpu or config.get('use_gpu', False)
    setup_environment(use_gpu)

    original_dir = args.original_dir or config.get('original_dir', 'data/original_videos')
    sora_dir = args.sora_dir or config.get('sora_dir', 'data/sora_videos')
    output_dir = args.output_dir or config.get('output_dir', 'data/output')

    if args.max_frames:
        config['max_frames'] = args.max_frames
    if args.use_lpips:
        config['use_lpips'] = True
    if args.disable_object_detection:
        config['use_object_detection'] = False

    for directory in [original_dir, sora_dir, output_dir]:
        os.makedirs(directory, exist_ok=True)

    bias_categories = config.get('bias_categories', {
        'gender': ['male', 'female', 'nonbinary'],
        'ethnicity': ['asian', 'black', 'hispanic', 'white'],
        'age': ['child', 'young', 'adult', 'elderly'],
        'setting': ['urban', 'rural', 'indoor', 'outdoor'],
        'action': ['walking', 'running', 'sitting', 'standing']
    })

    pipeline = VideoComparisonPipeline(
        original_dir=original_dir,
        sora_dir=sora_dir,
        output_dir=output_dir,
        config=config
    )

    if args.pair:
        original_video, sora_video = args.pair
        pipeline.custom_pairs = [
            {
                "original": os.path.join(original_dir, original_video) if not os.path.isabs(
                    original_video) else original_video,
                "sora": os.path.join(sora_dir, sora_video) if not os.path.isabs(sora_video) else sora_video
            }
        ]
    else:
        pipeline.custom_pairs = [
            {
                "original": os.path.join(original_dir, "Seef_Heritage.mp4"),
                "sora": os.path.join(sora_dir, "Sora example.mp4")
            }
        ]

    try:
        results = pipeline.run_pipeline(bias_categories)
        print_analysis_summary(results)

    except Exception as e:
        import traceback
        print(f"Error running the analysis pipeline: {str(e)}")
        traceback.print_exc()
        print("\nPlease try running with --disable-object-detection if the error persists.")

if __name__ == "__main__":
    main()
