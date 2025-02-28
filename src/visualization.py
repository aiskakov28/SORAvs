"""
Enhanced visualization utilities for Sora video comparison results.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cv2
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker

class VisualizationGenerator:
    def __init__(self, output_dir, dpi=300, color_palette='coolwarm'):
        self.output_dir = output_dir
        self.dpi = dpi
        self.color_palette = color_palette

        os.makedirs(output_dir, exist_ok=True)

        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)

        self.quality_cmap = LinearSegmentedColormap.from_list(
            "quality", ["#FF4136", "#FFDC00", "#2ECC40"]
        )

    def create_metric_distributions(self, df):
        plt.figure(figsize=(16, 12))

        metrics = [
            ('structural_similarity', 'Structural Similarity (SSIM)', 'blue', True),
            ('feature_similarity', 'Feature Distance', 'green', False),
            ('motion_similarity', 'Motion Difference', 'orange', False),
            ('color_distribution', 'Color Histogram Distance', 'red', False),
            ('lpips_score', 'LPIPS Perceptual Distance', 'purple', False),
            ('object_accuracy', 'Object Recognition Accuracy', 'brown', True)
        ]

        for i, (col, title, color, higher_better) in enumerate(metrics):
            if col in df.columns:
                plt.subplot(3, 2, i+1)

                sns.histplot(df[col], kde=True, color=color)

                min_val = df[col].min()
                max_val = df[col].max()
                avg_val = df[col].mean()

                if higher_better:
                    quality_text = f"Higher is better | Avg: {avg_val:.3f} | Range: [{min_val:.3f} - {max_val:.3f}]"
                else:
                    quality_text = f"Lower is better | Avg: {avg_val:.3f} | Range: [{min_val:.3f} - {max_val:.3f}]"

                plt.title(f"Distribution of {title}\n{quality_text}")
                plt.xlabel(title)
                plt.ylabel("Frequency")

                plt.axvline(avg_val, color='black', linestyle='--', alpha=0.7)
                plt.text(avg_val, plt.gca().get_ylim()[1]*0.9, f'Mean: {avg_val:.3f}',
                        horizontalalignment='center', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_distributions.png'), dpi=self.dpi)
        plt.close()

    def create_correlation_matrix(self, df):
        numeric_df = df.select_dtypes(include=[np.number])

        plt.figure(figsize=(12, 10))

        col_mapping = {
            'structural_similarity': 'SSIM',
            'feature_similarity': 'Feature Dist',
            'motion_similarity': 'Motion Diff',
            'color_distribution': 'Color Dist',
            'lpips_score': 'LPIPS',
            'object_accuracy': 'Object Acc',
            'perceived_quality': 'Perc Quality',
            'temporal_consistency': 'Temp Consist'
        }

        numeric_df = numeric_df.rename(columns={col: col_mapping.get(col, col) for col in numeric_df.columns if col in col_mapping})

        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        cmap = LinearSegmentedColormap.from_list("custom", ["#4169E1", "#FFFFFF", "#FF6347"])

        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .8})

        plt.title('Correlation Between Metrics', fontsize=16, pad=20)
        plt.figtext(0.5, 0.01,
                   "Positive correlation (red): Metrics increase together\n"
                   "Negative correlation (blue): As one metric increases, the other decreases\n"
                   "No correlation (white): Metrics vary independently",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'metric_correlations.png'), dpi=self.dpi)
        plt.close()

    def create_bias_category_plots(self, bias_metrics):
        for category, values in bias_metrics.items():
            if not values:
                continue

            plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 2, figure=plt.gcf())

            labels = list(values.keys())

            metrics_to_plot = [
                ('avg_ssim', 'SSIM Score', True, 0),
                ('avg_perceived_quality', 'Perceived Quality', True, 0),
                ('avg_feature_distance', 'Feature Distance', False, 2),
                ('avg_lpips', 'LPIPS Distance', False, 2),
                ('avg_objects_accuracy', 'Object Recognition Accuracy', True, 3),
                ('avg_temporal_consistency', 'Temporal Consistency', True, 3)
            ]

            for i, (metric_key, metric_name, higher_better, pos) in enumerate(metrics_to_plot):
                if all(metric_key in values[label] for label in labels):
                    metric_values = [values[label][metric_key] for label in labels]

                    ax = plt.subplot(gs[pos])

                    if higher_better:
                        colors = [self.quality_cmap(val) for val in np.array(metric_values)/max(metric_values) if max(metric_values) > 0]
                    else:
                        norm_vals = 1 - (np.array(metric_values)/max(metric_values) if max(metric_values) > 0 else np.zeros_like(metric_values))
                        colors = [self.quality_cmap(val) for val in norm_vals]

                    bars = ax.bar(labels, metric_values, color=colors)

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(metric_values),
                               f'{height:.3f}', ha='center', va='bottom', fontsize=10)

                    direction = "Higher is better" if higher_better else "Lower is better"
                    ax.set_title(f'{metric_name} by {category.capitalize()}\n({direction})')
                    ax.set_ylabel(metric_name)

                    if len(labels) > 4:
                        plt.xticks(rotation=45, ha='right')

            ax = plt.subplot(gs[4:])

            norm_metrics = {}
            for metric_key, metric_name, higher_better, _ in metrics_to_plot:
                if all(metric_key in values[label] for label in labels):
                    raw_values = [values[label][metric_key] for label in labels]

                    if higher_better:
                        if max(raw_values) > 0:
                            norm_values = [val/max(raw_values) for val in raw_values]
                        else:
                            norm_values = [0.0] * len(raw_values)
                    else:
                        if max(raw_values) > 0:
                            norm_values = [1 - (val/max(raw_values)) for val in raw_values]
                        else:
                            norm_values = [1.0] * len(raw_values)

                    norm_metrics[metric_name] = norm_values

            if norm_metrics:
                metric_names = list(norm_metrics.keys())
                x = np.arange(len(labels))
                width = 0.8 / len(metric_names)

                for i, metric_name in enumerate(metric_names):
                    offset = (i - len(metric_names)/2 + 0.5) * width
                    ax.bar(x + offset, norm_metrics[metric_name], width, label=metric_name)

                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_title(f'Normalized Metrics by {category.capitalize()}\n(Higher is better)')
                ax.set_ylabel('Normalized Score (0-1)')
                ax.set_ylim(0, 1.1)
                ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

                if len(labels) > 4:
                    plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'bias_{category}_comprehensive.png'), dpi=self.dpi)
            plt.close()

    def create_single_video_comparison(self, original_frames, sora_frames, metrics, output_prefix):
        min_frames = min(len(original_frames), len(sora_frames))
        if min_frames == 0:
            return

        if min_frames >= 5:
            indices = [0, min_frames // 4, min_frames // 2, min_frames * 3 // 4, min_frames - 1]
        elif min_frames >= 3:
            indices = [0, min_frames // 2, min_frames - 1]
        else:
            indices = list(range(min_frames))

        fig = plt.figure(figsize=(15, 5 * len(indices)))
        gs = GridSpec(len(indices), 3, figure=fig, width_ratios=[1, 1, 0.5])

        for i, idx in enumerate(indices):
            ax1 = plt.subplot(gs[i, 0])
            ax1.imshow(original_frames[idx])
            ax1.set_title(f'Original Frame {idx}')
            ax1.axis('off')

            ax2 = plt.subplot(gs[i, 1])
            ax2.imshow(sora_frames[idx])
            ax2.set_title(f'Sora Frame {idx}')
            ax2.axis('off')

            ax3 = plt.subplot(gs[i, 2])
            if original_frames[idx].shape == sora_frames[idx].shape:
                diff = np.abs(original_frames[idx].astype(float) - sora_frames[idx].astype(float))
                if diff.max() > 0:
                    diff = diff / diff.max()
                ax3.imshow(diff, cmap='hot')
                ax3.set_title(f'Difference Map')
                ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, "Frame sizes don't match", ha='center', va='center')
                ax3.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_frame_comparison.png'), dpi=self.dpi)
        plt.close()
        plt.figure(figsize=(12, 10))

        normalized_metrics = metrics.get('normalized_metrics', {})
        metric_names = list(normalized_metrics.keys())
        metric_values = [normalized_metrics.get(name, 0) for name in metric_names]
        ax = plt.subplot(121, polar=True)
        angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()

        metric_values += metric_values[:1]
        angles += angles[:1]
        metric_names += ['']

        ax.plot(angles, metric_values, 'o-', linewidth=2, label='Normalized Metrics')
        ax.fill(angles, metric_values, alpha=0.25)

        ax.set_thetagrids(np.degrees(angles[:-1]), metric_names[:-1])

        ax.set_ylim(0, 1)
        ax.set_title('Normalized Similarity Metrics\n(1 = Perfect Match)')

        ax2 = plt.subplot(122)
        raw_metrics = metrics.get('raw_metrics', {})

        display_metrics = [
            ('ssim', 'SSIM', 0, 1, True),
            ('lpips', 'LPIPS', 0, 0.5, False),
            ('feature_distance', 'Feature Dist', 0, 5000, False),
            ('motion_difference', 'Motion Diff', 0, 5, False),
            ('color_difference', 'Color Diff', 0, 2, False),
            ('objects_accuracy', 'Object Acc', 0, 1, True)
        ]

        metric_keys = []
        metric_display = []
        metric_vals = []
        colors = []

        for key, display, min_range, max_range, higher_better in display_metrics:
            if key in raw_metrics:
                metric_keys.append(key)
                metric_display.append(display)
                val = raw_metrics[key]
                metric_vals.append(val)

                if higher_better:
                    norm_val = (val - min_range) / (max_range - min_range) if max_range > min_range else 0.5
                else:
                    norm_val = 1 - ((val - min_range) / (max_range - min_range) if max_range > min_range else 0.5)
                norm_val = max(0, min(1, norm_val))
                colors.append(self.quality_cmap(norm_val))

        bars = ax2.bar(metric_display, metric_vals, color=colors)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')

        ax2.set_title('Raw Metrics')
        if max(metric_vals) / (min(metric_vals) + 1e-10) > 100:
            ax2.set_yscale('log')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_metrics_summary.png'), dpi=self.dpi)
        plt.close()

        if 'frame_metrics' in metrics and any(metrics['frame_metrics']):
            self.create_frame_by_frame_comparison(
                metrics['frame_metrics'].get('ssim_scores', []),
                metrics['frame_metrics'].get('feature_distances', []),
                metrics['frame_metrics'].get('lpips_scores', []),
                output_prefix
            )

    def create_frame_by_frame_comparison(self, ssim_scores, feature_distances, lpips_scores, output_prefix):
        if not any([ssim_scores, feature_distances, lpips_scores]):
            return

        plt.figure(figsize=(14, 10))
        max_frames = max(len(ssim_scores), len(feature_distances), len(lpips_scores))
        frame_indices = list(range(max_frames))

        if ssim_scores:
            plt.subplot(3, 1, 1)
            plt.plot(frame_indices[:len(ssim_scores)], ssim_scores, marker='o', markersize=3,
                    label='SSIM Score', color='blue')
            plt.axhline(np.mean(ssim_scores), color='blue', linestyle='--', alpha=0.7,
                       label=f'Mean: {np.mean(ssim_scores):.3f}')
            plt.fill_between(frame_indices[:len(ssim_scores)],
                            [np.mean(ssim_scores) - np.std(ssim_scores)] * len(ssim_scores),
                            [np.mean(ssim_scores) + np.std(ssim_scores)] * len(ssim_scores),
                            alpha=0.2, color='blue')
            plt.title('Frame-by-Frame SSIM Scores')
            plt.xlabel('Frame Number')
            plt.ylabel('SSIM Score')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()

        if feature_distances:
            plt.subplot(3, 1, 2)
            plt.plot(frame_indices[:len(feature_distances)], feature_distances, marker='o', markersize=3,
                    label='Feature Distance', color='green')
            plt.axhline(np.mean(feature_distances), color='green', linestyle='--', alpha=0.7,
                       label=f'Mean: {np.mean(feature_distances):.3f}')
            plt.title('Frame-by-Frame Feature Distances')
            plt.xlabel('Frame Number')
            plt.ylabel('Feature Distance')
            plt.grid(True)
            plt.legend()

        if lpips_scores:
            plt.subplot(3, 1, 3)
            plt.plot(frame_indices[:len(lpips_scores)], lpips_scores, marker='o', markersize=3,
                    label='LPIPS Distance', color='purple')
            plt.axhline(np.mean(lpips_scores), color='purple', linestyle='--', alpha=0.7,
                       label=f'Mean: {np.mean(lpips_scores):.3f}')
            plt.title('Frame-by-Frame LPIPS Perceptual Distances')
            plt.xlabel('Frame Number')
            plt.ylabel('LPIPS Distance')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_frame_metrics.png'), dpi=self.dpi)
        plt.close()

        if max_frames > 1:
            available_metrics = []
            if ssim_scores:
                available_metrics.append(('SSIM', ssim_scores, True))  # True means higher is better
            if feature_distances:
                available_metrics.append(('Feature', feature_distances, False))  # False means lower is better
            if lpips_scores:
                available_metrics.append(('LPIPS', lpips_scores, False))

            if available_metrics:
                plt.figure(figsize=(max(8, max_frames/5), len(available_metrics) * 2))
                data = []
                ylabels = []

                for name, values, higher_better in available_metrics:
                    if higher_better:
                        norm_values = values / np.max(values) if np.max(values) > 0 else values
                    else:
                        norm_values = 1 - (values / np.max(values) if np.max(values) > 0 else values)
                    data.append(norm_values)
                    ylabels.append(name)

                plt.imshow(data, aspect='auto', cmap=self.quality_cmap)
                plt.colorbar(label='Normalized Quality (1 = Best)')
                plt.yticks(range(len(ylabels)), ylabels)

                if max_frames > 20:
                    step = max(1, max_frames // 20)
                    plt.xticks(range(0, max_frames, step))
                else:
                    plt.xticks(range(max_frames))

                plt.xlabel('Frame Number')
                plt.title('Frame-by-Frame Quality Heatmap')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_quality_heatmap.png'), dpi=self.dpi)
                plt.close()

    def create_object_detection_visualization(self, original_frame, sora_frame, original_objects, sora_objects, output_prefix, frame_idx):
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(original_frame)
        ax = plt.gca()

        for obj in original_objects:
            x1, y1, x2, y2 = obj['box']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f"{obj['class']}: {obj['confidence']:.2f}",
                    color='white', bbox=dict(facecolor='red', alpha=0.5))

        plt.title(f'Original Frame {frame_idx}: {len(original_objects)} Objects Detected')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(sora_frame)
        ax = plt.gca()

        for obj in sora_objects:
            x1, y1, x2, y2 = obj['box']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f"{obj['class']}: {obj['confidence']:.2f}",
                    color='white', bbox=dict(facecolor='red', alpha=0.5))
        plt.title(f'Sora Frame {frame_idx}: {len(sora_objects)} Objects Detected')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_objects_frame{frame_idx}.png'), dpi=self.dpi)
        plt.close()

    def create_depth_map_comparison(self, original_depth, sora_depth, output_prefix, frame_idx):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_depth, cmap='viridis')
        plt.colorbar(label='Depth')
        plt.title(f'Original Frame {frame_idx} Depth Map')

        plt.subplot(1, 2, 2)
        plt.imshow(sora_depth, cmap='viridis')
        plt.colorbar(label='Depth')
        plt.title(f'Sora Frame {frame_idx} Depth Map')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_depth_frame{frame_idx}.png'), dpi=self.dpi)
        plt.close()

    def create_scene_transition_comparison(self, original_scenes, sora_scenes, original_duration, sora_duration, output_prefix):
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        for i, (start, end) in enumerate(original_scenes):
            norm_start = start / original_duration
            norm_end = end / original_duration
            plt.barh(0, norm_end - norm_start, left=norm_start, height=0.5,
                   color=plt.cm.tab10(i % 10), alpha=0.7)
            plt.text(norm_start + (norm_end - norm_start)/2, 0, f'Scene {i+1}',
                   ha='center', va='center')

        plt.title('Original Video Scene Timeline')
        plt.xlim(0, 1)
        plt.xticks([i/10 for i in range(11)], [f'{i*10}%' for i in range(11)])
        plt.yticks([])
        plt.xlabel('Normalized Video Duration')

        plt.subplot(2, 1, 2)
        for i, (start, end) in enumerate(sora_scenes):
            norm_start = start / sora_duration
            norm_end = end / sora_duration
            plt.barh(0, norm_end - norm_start, left=norm_start, height=0.5,
                   color=plt.cm.tab10(i % 10), alpha=0.7)
            plt.text(norm_start + (norm_end - norm_start)/2, 0, f'Scene {i+1}',
                   ha='center', va='center')

        plt.title('Sora Video Scene Timeline')
        plt.xlim(0, 1)
        plt.xticks([i/10 for i in range(11)], [f'{i*10}%' for i in range(11)])
        plt.yticks([])
        plt.xlabel('Normalized Video Duration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_scene_transitions.png'), dpi=self.dpi)
        plt.close()