#!/usr/bin/env python3
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import multiprocessing

import numpy as np
import torch
from PIL import Image, ExifTags
from torchvision import transforms

# ONNX Runtime imports - you'll need to install this package
import onnxruntime as ort

# Define the transformation for image processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ProgressTracker:
    """Tracks and logs progress at regular intervals."""

    def __init__(self, total_items, log_interval=3.0):
        """
        Initialize the progress tracker.

        Args:
            total_items: Total number of items to process
            log_interval: Time interval in seconds between logs
        """
        self.total_items = total_items
        self.processed_items = 0
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.lock = Lock()  # For thread safety

        # Initial empty progress bar
        print(f"Progress: 0.00% (0/{total_items})")

    def update(self, items=1):
        """
        Update progress and log if necessary.

        Args:
            items: Number of items processed in this update
        """
        with self.lock:
            self.processed_items += items
            current_time = time.time()

            # Log progress if log_interval has passed
            if current_time - self.last_log_time >= self.log_interval:
                self._log_progress()
                self.last_log_time = current_time

    def _log_progress(self):
        """Log current progress."""
        percentage = (self.processed_items / self.total_items) * 100
        elapsed_time = time.time() - self.start_time

        # Calculate estimated time remaining
        if self.processed_items > 0:
            items_per_second = self.processed_items / elapsed_time
            estimated_remaining = (self.total_items - self.processed_items) / items_per_second

            time_str = f", ETA: {format_time(estimated_remaining)}"
        else:
            time_str = ""

        print(f"Progress: {percentage:.2f}% ({self.processed_items}/{self.total_items}{time_str})")

    def finish(self):
        """Mark processing as complete and log final progress."""
        elapsed_time = time.time() - self.start_time
        print(f"Completed: 100% ({self.total_items}/{self.total_items})")
        print(f"Total time: {format_time(elapsed_time)}")


def format_time(seconds):
    """Format seconds into a human-readable time string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


class OrientationDetectionONNX:
    """Detect the orientation of images using an ONNX model converted from ResNet152."""
    angles = [0, 90, 180, 270]  # possible orientation classes

    def __init__(self, onnx_path: str, device: str = 'cpu', suppress_warnings: bool = False,
                 verbose: bool = False):
        """Initialize the ONNX orientation detection model.

        Args:
            onnx_path: Path to the ONNX model
            device: Device to use for inference ('cuda' or 'cpu')
            suppress_warnings: Whether to suppress consistency warnings
            verbose: Whether to show verbose output
        """
        self.device = device
        self.onnx_path = onnx_path
        self.suppress_warnings = suppress_warnings
        self.verbose = verbose

        # Select appropriate provider based on device
        providers = ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider'] + providers
            print("Using CUDA with ONNX Runtime")
        else:
            print("Using CPU with ONNX Runtime")

        # Create ONNX Runtime session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Enable optimizations specific to CPU if using CPU
        if device == 'cpu':
            sess_options.add_session_config_entry('session.set_denormal_as_zero', '1')
            sess_options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def read_image(image_path: str) -> Image.Image:
        """Read an image from disk and convert to RGB."""
        return Image.open(image_path).convert("RGB")

    def to_numpy(self, image: Image) -> np.ndarray:
        """Convert PIL image to numpy array for model input."""
        image = image.convert("RGB")
        tensor_image = transform(image).unsqueeze(0)
        return tensor_image.numpy()

    def get_angles(self, numpy_image: np.ndarray) -> dict:
        """Get orientation probabilities for a numpy image."""
        # Perform inference
        outputs = self.session.run(None, {self.input_name: numpy_image})

        # Apply softmax to outputs to get probabilities
        scores = outputs[0][0]
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

        angles = {angle: score for angle, score in zip(self.angles, probabilities)}
        return angles

    def get_angles_avg(self, image: Image.Image) -> tuple:
        """
        Process an image with all four rotations and average the predictions,
        accounting for the rotations using improved consistency checking.

        Returns:
            tuple: (angles_dict, inconsistencies, confidence_score)
        """
        # Convert image to numpy (matches the ONNX input format)
        numpy_image = self.to_numpy(image)

        # Create a dictionary to accumulate scores for each absolute orientation
        accumulated = {a: [] for a in self.angles}

        # Track predictions for each rotation for analysis
        rotation_predictions = []

        # Track inconsistencies
        inconsistencies = []

        # For each rotation (0°, 90°, 180°, 270°)
        for rotation_idx, rotation in enumerate(self.angles):
            # Get predictions for the current orientation
            angles = self.get_angles(numpy_image.copy())

            # Store prediction details for this rotation
            prediction_details = {
                'rotation': rotation,
                'predicted_angles': angles,
                'best_angle': max(angles, key=angles.get),
                'best_score': angles[max(angles, key=angles.get)]
            }
            rotation_predictions.append(prediction_details)

            # Adjust angles to absolute orientation (relative to original image)
            for pred_angle, score in angles.items():
                # The formula: absolute_angle = (pred_angle - rotation) % 360
                absolute_angle = (pred_angle - rotation) % 360
                accumulated[absolute_angle].append(score)

            # Rotate the numpy array for the next iteration
            numpy_image = np.rot90(numpy_image, k=1, axes=(2, 3)).copy()

        # Run improved consistency analysis on the collected predictions
        inconsistencies, reliability_score = self._analyze_rotation_consistency(rotation_predictions)

        # Average the scores for each absolute orientation, with potential weighting
        result = {}
        for angle in sorted(accumulated.keys()):
            # Basic approach: simple average
            overall_avg = sum(accumulated[angle]) / len(accumulated[angle])
            result[angle] = overall_avg

        return result, inconsistencies, reliability_score

    def _analyze_rotation_consistency(self, predictions):
        """
        Analyze the consistency of predictions across different rotations with
        more balanced detection criteria.

        Args:
            predictions: List of dictionaries with prediction details for each rotation

        Returns:
            tuple: (inconsistencies, reliability_score)
        """
        inconsistencies = []

        # Extract base prediction (from original orientation)
        base_pred = predictions[0]
        base_angle = base_pred['best_angle']
        base_confidence = base_pred['best_score']

        # Calculate first/second confidence ratio for each rotation
        for pred in predictions:
            sorted_scores = sorted(pred['predicted_angles'].values(), reverse=True)
            if len(sorted_scores) >= 2:
                pred['confidence_ratio'] = sorted_scores[0] / max(sorted_scores[1], 0.001)  # Avoid div by zero
            else:
                pred['confidence_ratio'] = float('inf')  # Very high ratio if only one score

        # Calculate maximum confidence for any prediction
        max_confidence = max(pred['best_score'] for pred in predictions)

        # Check consistency for each rotated prediction
        num_inconsistencies = 0
        rotation_issues = {}  # Track issues by rotation for filtering

        for i, pred in enumerate(predictions[1:], 1):  # Skip the base prediction
            rotation = pred['rotation']
            actual_angle = pred['best_angle']
            expected_angle = (base_angle + rotation) % 360

            # Calculate key metrics
            confidence = pred['best_score']
            expected_confidence = pred['predicted_angles'].get(expected_angle, 0)
            confidence_diff = abs(confidence - expected_confidence)

            # MODIFIED: More conservative adaptive threshold
            base_threshold = 0.4  # Increased from 0.3 to 0.4
            confidence_factor = 0.5 + 0.5 * (base_confidence + confidence) / 2  # Scale from 0.5 to 1.0
            adaptive_threshold = base_threshold * (2 - confidence_factor)  # Lower threshold for higher confidence

            # Check if this is an inconsistency using more balanced criteria
            is_inconsistent = False
            reason = ""

            # MODIFIED: More selective criteria
            # Criterion 1: Very significant confidence difference with high confidence predictions
            if confidence_diff > adaptive_threshold and confidence > 0.5 and expected_confidence > 0.2:
                is_inconsistent = True
                reason = f"large confidence difference ({confidence_diff:.2f} > {adaptive_threshold:.2f})"

            # Criterion 2: Completely wrong angle prediction with very high confidence
            elif actual_angle != expected_angle and confidence > 0.7 and base_confidence > 0.7:
                # Only consider very large angle differences
                angle_diff = min((actual_angle - expected_angle) % 360, (expected_angle - actual_angle) % 360)
                if angle_diff >= 180:  # Only the most extreme cases
                    is_inconsistent = True
                    reason = f"opposite angle prediction with high confidence"

            # REMOVED: Low confidence ratio criterion (too sensitive)

            if is_inconsistent and not self.suppress_warnings:
                rotation_issues[rotation] = {
                    'message': f"Rotation consistency check - Expected {expected_angle}° but got {actual_angle}° "
                               f"(scores: {confidence:.2f} vs {expected_confidence:.2f}), {reason}",
                    'severity': confidence_diff,  # Use difference as severity
                    'rotation': rotation
                }
                num_inconsistencies += 1

        # ADDED: Filter to include only the most significant inconsistencies
        # This prevents counting minor issues while keeping the important ones
        if rotation_issues:
            # Sort by severity and take only the top 2 most significant issues
            significant_issues = sorted(rotation_issues.values(), key=lambda x: x['severity'], reverse=True)[:2]
            for issue in significant_issues:
                inconsistencies.append(issue['message'])

        # Calculate an improved reliability score (0.0 to 1.0)
        # Less sensitive to minor inconsistencies
        if len(predictions) <= 1:
            reliability_score = base_confidence  # Just use base confidence if only one prediction
        else:
            # Start with average of top confidences
            avg_confidence = sum(pred['best_score'] for pred in predictions) / len(predictions)

            # Penalize for inconsistencies, but less severely
            # MODIFIED: More forgiving consistency factor
            consistency_factor = max(0, 1 - (num_inconsistencies / (len(predictions) * 2)))

            # Consider confidence ratio (higher is better)
            avg_ratio = sum(pred.get('confidence_ratio', 1) for pred in predictions) / len(predictions)
            ratio_factor = min(1.0, avg_ratio / 3)  # Cap at 1.0, with 3.0 being "ideal"

            # MODIFIED: Changed weights to emphasize confidence more
            reliability_score = avg_confidence * (0.6 + 0.3 * consistency_factor + 0.1 * ratio_factor)

        return inconsistencies, reliability_score

    def get_best_angle(self, image: Image.Image) -> tuple:
        """
        Get the best angle and its confidence using the improved averaged method.
        Returns (best_angle, confidence, all_probabilities, inconsistencies, reliability)
        """
        angles, inconsistencies, reliability = self.get_angles_avg(image)
        best_angle = max(angles, key=angles.get)
        return best_angle, angles[best_angle], angles, inconsistencies, reliability


class ImageProcessor:
    """Processes images for orientation detection and correction."""

    @staticmethod
    def preserve_file_metadata(source_path, target_path):
        """
        Preserve the file's metadata (creation time, modification time, etc.)

        Args:
            source_path: Path to the original file
            target_path: Path to the new file
        """
        # Get original file stats
        src_stat = os.stat(source_path)

        # Preserve access and modification times
        os.utime(target_path, (src_stat.st_atime, src_stat.st_mtime))

    @staticmethod
    def correct_orientation(img, best_angle, exif_data):
        """
        Correct the orientation of an image.

        Args:
            img: PIL Image to correct
            best_angle: Angle to rotate (90, 180, or 270 degrees)
            exif_data: EXIF data from the original image

        Returns:
            PIL.Image: Corrected image
        """
        # Determine rotation method
        rotation_methods = {
            90: Image.Transpose.ROTATE_270,  # Counterclockwise to correct 90° clockwise
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_90,  # Clockwise to correct 270° clockwise
        }

        # Rotate the image
        corrected_img = img.transpose(rotation_methods[best_angle])

        # Copy EXIF data to the rotated image
        if exif_data:
            # Make sure to update the orientation tag to normal (1)
            # Find the orientation tag
            orientation_tag = None
            for tag, tag_value in ExifTags.TAGS.items():
                if tag_value == 'Orientation':
                    orientation_tag = tag
                    break

            if orientation_tag and orientation_tag in exif_data:
                exif_data[orientation_tag] = 1  # Normal orientation

            # Apply the EXIF data to the corrected image
            corrected_img.info["exif"] = exif_data.tobytes()

        return corrected_img

    @staticmethod
    def save_image(img, out_path, exif_data):
        """
        Save an image to disk.

        Args:
            img: PIL Image to save
            out_path: Path to save the image to
            exif_data: EXIF data to include in the saved image
        """
        # Create the parent directory if it doesn't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the image with original EXIF data
        img.save(out_path, exif=exif_data)


def process_image_batch(batch, onnx_path, args, output_folder, process_id=0):
    """
    Process a batch of images with improved correction decisions.

    Args:
        batch: List of image paths to process
        onnx_path: Path to the ONNX model
        args: Command line arguments
        output_folder: Output folder for corrected images
        process_id: ID of the current process (for logging)

    Returns:
        list: Results of the processing
    """
    # Initialize ONNX model for this process
    od = OrientationDetectionONNX(
        onnx_path=onnx_path,
        device=args.device,
        suppress_warnings=args.suppress_warnings,
        verbose=args.verbose
    )

    results = []

    for i, img_path in enumerate(batch):
        result = {
            'img_path': img_path,
            'best_angle': None,
            'confidence': 0,
            'all_probs': {},
            'corrected': False,
            'status': 'Processed',
            'inconsistencies': [],
            'reliability': 0,
            'error': None
        }

        try:
            # Load the image
            img = od.read_image(str(img_path))

            # Get the original image's EXIF data
            try:
                exif_data = img.getexif()
            except Exception:
                exif_data = None

            # Get the best angle, confidence, all probabilities, inconsistencies, and reliability
            best_angle, confidence, all_probs, inconsistencies, reliability = od.get_best_angle(img)

            # Update results
            result['best_angle'] = best_angle
            result['confidence'] = confidence
            result['all_probs'] = all_probs
            result['inconsistencies'] = inconsistencies
            result['reliability'] = reliability

            # Calculate secondary metrics for decision-making
            second_best_angle = None
            second_best_conf = 0
            for angle, conf in all_probs.items():
                if angle != best_angle and conf > second_best_conf:
                    second_best_angle = angle
                    second_best_conf = conf

            # Confidence margin is the ratio between best and second best
            confidence_margin = confidence / max(second_best_conf, 0.001)  # Avoid div by zero

            # Determine if we should trust this prediction for correction
            should_correct = False
            correction_reason = "insufficient confidence"

            # Base threshold from args
            base_threshold = args.threshold

            # NEW: Simplified and more balanced decision logic
            # 1. If no inconsistencies and good confidence, correct
            if not inconsistencies and confidence > base_threshold * 0.9:
                should_correct = True
                correction_reason = "good confidence, no inconsistencies"

            # 2. Very high confidence overrides minor inconsistencies
            elif confidence > base_threshold + 0.05 and confidence_margin > 3.5:
                should_correct = True
                correction_reason = "very high confidence margin"

            # 3. Good reliability is more important than inconsistencies
            elif reliability > 0.75:
                should_correct = True
                correction_reason = f"strong reliability ({reliability:.2f})"

            # 4. For non-zero angles with dominant confidence, correct regardless
            elif best_angle != 0 and confidence > 0.75 and second_best_conf < 0.15:
                should_correct = True
                correction_reason = "clear dominant orientation"

            # NEW: Add visualization about confidence distribution
            prefix = f"[Process {process_id}, Item {i}] "
            if args.verbose:
                print(f"\n{prefix}{img_path}:")
                print(f"Reliability: {reliability:.2f}, Confidence margin: {confidence_margin:.2f}")

                # Create a simple ASCII visualization of confidence distribution
                print("Confidence distribution:")
                max_bars = 40  # Maximum bar length
                for angle, prob in sorted(all_probs.items()):
                    bar_length = int(prob * max_bars)
                    bar = "█" * bar_length
                    star = "*" if angle == best_angle else " "
                    print(f"{star} {angle:03d}°: {prob * 100:5.1f}% |{bar}")

                if should_correct:
                    print(f"Decision: CORRECT ({correction_reason})")
                else:
                    print(f"Decision: SKIP ({correction_reason})")
            else:
                print(
                    f"{prefix}{img_path}: Orientation {best_angle:03d}° (Confidence: {confidence * 100:.1f}%, Reliability: {reliability * 100:.1f}%)")

            # Correct the orientation if requested and our decision logic says we should
            if args.correct and best_angle != 0 and should_correct:
                # Correct the image
                corrected_img = ImageProcessor.correct_orientation(img, best_angle, exif_data)

                # Determine the output path
                if output_folder:
                    # Preserve the directory structure relative to input_folder
                    rel_path = img_path.relative_to(args.input_folder)
                    out_path = output_folder / rel_path
                else:
                    out_path = img_path

                # Save the corrected image
                ImageProcessor.save_image(corrected_img, out_path, exif_data)

                # Preserve file metadata
                ImageProcessor.preserve_file_metadata(str(img_path), str(out_path))

                result['corrected'] = True
                result['status'] = f"Corrected ({correction_reason})"
                print(f"  - Corrected to 0° and saved to {out_path} ({correction_reason})")
            elif args.correct and best_angle != 0:
                print(f"  - Not corrected: {correction_reason}")

        except Exception as e:
            result['status'] = 'Error'
            result['error'] = str(e)
            print(f"Error processing {img_path}: {e}")

        results.append(result)

    return results


def csv_writer_process(csv_path, result_queue, total_images):
    """
    Separate process to write results to CSV file.

    Args:
        csv_path: Path to the CSV file
        result_queue: Queue to receive results from worker processes
        total_images: Total number of images to process
    """
    if not csv_path:
        return

    try:
        # Open CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header with new reliability column
            writer.writerow([
                'Image', 'Best Angle', 'Confidence', 'Reliability',
                'Prob 0°', 'Prob 90°', 'Prob 180°', 'Prob 270°',
                'Corrected', 'Status', 'Inconsistencies'
            ])

            # Process results as they come in
            processed = 0
            while processed < total_images:
                result = result_queue.get()

                if result is None:
                    # End signal received
                    break

                # Write result to CSV
                inconsistencies_str = "; ".join(result['inconsistencies']) if result['inconsistencies'] else ""

                if result['status'] == 'Error':
                    writer.writerow([
                        str(result['img_path']),
                        "ERROR",
                        str(result['error']),
                        "",  # Reliability
                        "", "", "", "",  # Probabilities
                        "No",
                        "Error",
                        ""
                    ])
                else:
                    writer.writerow([
                        str(result['img_path']),
                        result['best_angle'],
                        f"{result['confidence'] * 100:.2f}%",
                        f"{result.get('reliability', 0) * 100:.2f}%",  # New reliability column
                        f"{result['all_probs'][0] * 100:.2f}%",
                        f"{result['all_probs'][90] * 100:.2f}%",
                        f"{result['all_probs'][180] * 100:.2f}%",
                        f"{result['all_probs'][270] * 100:.2f}%",
                        "Yes" if result['corrected'] else "No",
                        result['status'],
                        inconsistencies_str
                    ])

                processed += 1

                # Flush to disk periodically
                if processed % 10 == 0:
                    f.flush()

        print(f"Results saved to {csv_path}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")


def recursive_iterdir(path: Path):
    """Recursively iterate through a directory and yield all files."""
    path = Path(path)
    for i in path.iterdir():
        if i.is_dir():
            yield from recursive_iterdir(i)
        else:
            yield i


def main():
    parser = argparse.ArgumentParser(description='Batch image orientation detection')
    parser.add_argument('input_folder', type=str, help='Path to folder containing images')
    parser.add_argument('--model', type=str, default='resnet152_ixion_e3-fac493d9.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Confidence threshold for automatic correction (0-1)')
    parser.add_argument('--correct', action='store_true',
                        help='Automatically correct image orientation if confidence > threshold')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder for corrected images (if not specified, originals will be overwritten)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save results to CSV file')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to use for inference (cuda or cpu)')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of images to process in each batch')
    parser.add_argument('--suppress_warnings', action='store_true',
                        help='Suppress all consistency warnings')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed probabilities for each image')
    parser.add_argument('--log_interval', type=float, default=3.0,
                        help='Time interval in seconds between progress logs')
    parser.add_argument('--use_smart_correction', action='store_true', default=True,
                        help='Use smart correction logic instead of simple threshold')
    parser.add_argument('--skip_processed', action='store_true',
                        help='Skip files that have already been processed (requires --csv)')
    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Check if input folder exists
    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: Input folder '{input_folder}' not found or is not a directory.")
        return

    # Create output folder if specified and doesn't exist
    output_folder = None
    if args.output_folder:
        output_folder = Path(args.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set number of processes if not specified
    if args.num_processes is None:
        args.num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU for system

    print(f"PyTorch version: {torch.__version__}")

    # Supported image extensions
    Image.init()
    supported_extensions = {ext.lower() for ext in Image.EXTENSION}

    print(f"Scanning folder: {input_folder}")
    print(f"Confidence threshold: {args.threshold * 100:.1f}%")
    print(f"Using smart correction: {args.use_smart_correction}")

    # Find all image files
    image_files = []
    for path in recursive_iterdir(input_folder):
        if path.suffix.lower() in supported_extensions:
            image_files.append(path)

    print(f"Found {len(image_files)} image files")

    # Load already processed files if skip_processed is enabled
    if args.skip_processed and args.csv and Path(args.csv).exists():
        try:
            processed_files = set()
            with open(args.csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)

                for row in reader:
                    if row:
                        processed_files.add(row[0])

            # Filter out already processed files
            original_count = len(image_files)
            image_files = [img for img in image_files if str(img) not in processed_files]
            skipped_count = original_count - len(image_files)
            print(f"Skipping {skipped_count} already processed files")
        except Exception as e:
            print(f"Warning: Error reading CSV file: {e}")

    # Return early if no files to process
    if not image_files:
        print("No files to process")
        return

    # Create progress tracker
    progress_tracker = ProgressTracker(len(image_files), args.log_interval)

    # Create process pool
    print(f"==== batch processing starting with {args.num_processes} processes ====")

    # Create result queue for CSV writer if needed
    manager = multiprocessing.Manager()
    result_queue = manager.Queue() if args.csv else None

    # Start CSV writer process if needed
    csv_process = None
    if args.csv:
        csv_process = multiprocessing.Process(
            target=csv_writer_process,
            args=(args.csv, result_queue, len(image_files))
        )
        csv_process.start()

    # Divide images into equal batches for each process
    image_batches = []
    batch_size = max(1, len(image_files) // args.num_processes)

    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        image_batches.append(batch)

    # Create and start worker processes
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        # Submit all image batch processing tasks
        futures = []
        for i, batch in enumerate(image_batches):
            future = executor.submit(
                process_image_batch,
                batch,
                args.model,
                args,
                output_folder,
                i
            )
            futures.append((future, len(batch)))

        # Process results as they complete
        stats = {
            'processed': 0,
            'corrected': 0,
            'errors': 0,
            'inconsistent': 0
        }

        for future, batch_size in futures:
            try:
                batch_results = future.result()

                # Update statistics
                for result in batch_results:
                    if result['status'] == 'Error':
                        stats['errors'] += 1
                    elif result['corrected']:
                        stats['corrected'] += 1

                    if result['inconsistencies']:
                        stats['inconsistent'] += 1

                    # Send result to CSV writer if needed
                    if result_queue:
                        result_queue.put(result)

                # Update progress
                progress_tracker.update(batch_size)
                stats['processed'] += batch_size

            except Exception as e:
                print(f"Error processing batch: {e}")
                stats['errors'] += batch_size
                progress_tracker.update(batch_size)

    # Signal CSV writer to stop
    if result_queue:
        result_queue.put(None)

    # Wait for CSV writer to finish
    if csv_process:
        csv_process.join()

    # Mark progress as complete
    progress_tracker.finish()

    print("==== batch processing complete ====")
    print(f"Processed: {stats['processed']} images")
    print(f"Had Errors: {stats['errors']} images")
    print(f"Inconsistent: {stats['inconsistent']} images")

    if args.correct:
        print(f"Corrected {stats['corrected']} images")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Support for frozen executables
    main()