#!/usr/bin/env python3
import argparse
import csv
import os
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ExifTags
from torchvision import models, transforms

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
        accounting for the rotations.

        Returns:
            tuple: (angles_dict, inconsistencies)
        """
        # Convert image to numpy (matches the ONNX input format)
        numpy_image = self.to_numpy(image)

        # Create a dictionary to accumulate scores for each absolute orientation
        accumulated = {a: [] for a in self.angles}

        # Track inconsistencies
        inconsistencies = []

        # Base prediction (for the first orientation)
        base_pred = None

        # For each rotation (0°, 90°, 180°, 270°)
        for rotation in self.angles:
            # Get predictions for the current orientation
            angles = self.get_angles(numpy_image.copy())

            # Get the best angle prediction
            best_angle = max(angles, key=angles.get)

            # For the first rotation, save the baseline prediction
            if base_pred is None:
                base_pred = best_angle
            else:
                # Check that the model is consistent as we rotate
                expected_angle = (base_pred + rotation) % 360
                if best_angle != expected_angle:
                    # Only show consistency warnings in verbose mode or for significant discrepancies
                    best_score = angles[best_angle]
                    expected_score = angles.get(expected_angle, 0)
                    score_diff = abs(best_score - expected_score)

                    # Only warn if warnings aren't suppressed and there's a significant difference
                    if not self.suppress_warnings:
                        # 30% difference threshold and > 10% confidence?
                        if score_diff > 0.3 and best_score > 0.1 and expected_score > 0.1:
                            inconsistency_msg = (
                                    f"Rotation consistency check - Expected {expected_angle}° but got {best_angle}° " +
                                    f"(scores: {best_score:.2f} vs {expected_score:.2f})"
                            )
                            inconsistencies.append(inconsistency_msg)
                            print(f"Note: {inconsistency_msg}")

            # Adjust angles to absolute orientation (relative to original image)
            for pred_angle, score in angles.items():
                # The formula: absolute_angle = (pred_angle - rotation) % 360
                absolute_angle = (pred_angle - rotation) % 360
                accumulated[absolute_angle].append(score)

            # Rotate the numpy array for the next iteration using np.rot90
            # This matches how the ONNX implementation rotates
            # Use .copy() to create a contiguous array (fixes negative stride issues)
            numpy_image = np.rot90(numpy_image, k=1, axes=(2, 3)).copy()

        # Average the scores for each absolute orientation
        result = {}
        for angle in sorted(accumulated.keys()):
            overall_avg = sum(accumulated[angle]) / len(accumulated[angle])
            result[angle] = overall_avg

        return result, inconsistencies

    def get_best_angle(self, image: Image.Image) -> tuple:
        """
        Get the best angle and its confidence using the averaged method.
        Returns (best_angle, confidence, all_probabilities, inconsistencies)
        """
        angles, inconsistencies = self.get_angles_avg(image)
        best_angle = max(angles, key=angles.get)
        return best_angle, angles[best_angle], angles, inconsistencies


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
    Process a batch of images.

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

            # Get the best angle, confidence, all probabilities, and inconsistencies
            best_angle, confidence, all_probs, inconsistencies = od.get_best_angle(img)

            # Update results
            result['best_angle'] = best_angle
            result['confidence'] = confidence
            result['all_probs'] = all_probs
            result['inconsistencies'] = inconsistencies

            # Print the result if verbose or not part of a batch
            prefix = f"[Process {process_id}, Item {i}] "
            if args.verbose:
                print(f"\n{prefix}{img_path}:")
                for angle, prob in sorted(all_probs.items()):
                    star = "*" if angle == best_angle else " "
                    print(f"{star} {angle}°: {prob * 100:.2f}%")
            else:
                print(f"{prefix}{img_path}: Orientation {best_angle:03d}° (Confidence: {confidence * 100:.2f}%)")

            # Correct the orientation if requested and confidence is high enough
            if args.correct and confidence > args.threshold and best_angle != 0:
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
                result['status'] = 'Corrected'
                print(f"  - Corrected to 0° and saved to {out_path}")

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

            # Write header
            writer.writerow([
                'Image', 'Best Angle', 'Confidence',
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
                        "", "", "", "",
                        "No",
                        "Error",
                        ""
                    ])
                else:
                    writer.writerow([
                        str(result['img_path']),
                        result['best_angle'],
                        f"{result['confidence'] * 100:.2f}%",
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
    parser.add_argument('--no_onnx', action='store_true',
                        help='Do not convert to ONNX, use PyTorch model directly (slower)')
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