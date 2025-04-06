#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import csv
import sys

# Define the transformation for image processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class OrientationDetection:
    """Detect the orientation of images using a pre-trained ResNet152 model."""
    angles = [0, 90, 180, 270]  # possible orientation classes
    _model = None

    def __init__(self, checkpoint_path: str, device: str | None = None, suppress_warnings: bool = False,
                 force_gpu: bool = False, verbose: bool = False):
        """Initialize the orientation detection model.

        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to use for inference ('cuda' or 'cpu')
            suppress_warnings: Whether to suppress consistency warnings
            force_gpu: Whether to force GPU usage and raise error if not available
            verbose: Whether to show verbose output
        """
        # Check CUDA availability with diagnostic information
        cuda_available = torch.cuda.is_available()

        if device is None:
            if cuda_available:
                device = "cuda"
            elif force_gpu:
                raise RuntimeError("GPU usage forced but CUDA is not available")
            else:
                device = "cpu"

        # Print diagnostic information about CUDA
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            print(f"GPU device name: {torch.cuda.get_device_name(0)}")

        self.device = device
        self.checkpoint_path = checkpoint_path
        self.suppress_warnings = suppress_warnings
        self.verbose = verbose
        print(f"Using device: {device}")

    @staticmethod
    def read_image(image_path: str) -> Image.Image:
        """Read an image from disk and convert to RGB."""
        return Image.open(image_path).convert("RGB")

    @property
    def model(self):
        """Load the model if not already loaded."""
        if self._model is None:
            num_classes = len(self.angles)
            # Define the model architecture (ResNet152)
            # and update the final layer for classes
            model = models.resnet152(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            # Upload the model to device
            model.to(self.device)
            # Set the model to eval mode
            model.eval()
            self._model = model
        return self._model

    def to_tensor(self, image: Image) -> torch.Tensor:
        """Convert PIL image to tensor for model input."""
        image = image.convert("RGB")
        tensor_image = transform(image).unsqueeze(0).to(self.device)
        return tensor_image

    def get_angles(self, tensor_image: torch.Tensor) -> dict:
        """Get orientation probabilities for a tensor image."""
        # Perform inference and compute probabilities
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

            angles = {angle: score
                      for angle, score in zip(self.angles, probabilities)}
            return angles

    def get_angles_avg(self, image: Image.Image) -> dict:
        """
        Process an image with all four rotations and average the predictions,
        accounting for the rotations.
        """
        # Convert image to tensor first
        tensor_image = self.to_tensor(image)
        # Get the numpy array (CPU) - this matches the ONNX input format
        numpy_image = tensor_image.cpu().numpy()

        # Create a dictionary to accumulate scores for each absolute orientation
        accumulated = {a: [] for a in self.angles}

        # Base prediction (for the first orientation)
        base_pred = None

        # For each rotation (0°, 90°, 180°, 270°)
        for rotation in self.angles:
            # Make sure we have a contiguous array before converting to tensor
            current_tensor = torch.from_numpy(numpy_image.copy()).to(self.device)

            # Get predictions
            angles = self.get_angles(current_tensor)

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
                            print(
                                f"Note: Rotation consistency check - Expected {expected_angle}° but got {best_angle}° " +
                                f"(scores: {best_score:.2f} vs {expected_score:.2f})")

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

        return result

    def get_best_angle(self, image: Image.Image) -> tuple:
        """
        Get the best angle and its confidence using the averaged method.
        Returns (best_angle, confidence, all_probabilities)
        """
        angles = self.get_angles_avg(image)
        best_angle = max(angles, key=angles.get)
        return best_angle, angles[best_angle], angles


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
    parser.add_argument('--model', type=str, default='resnet152_ixion_e3-84529282.pth',
                        help='Path to the model checkpoint')
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
    parser.add_argument('--force_gpu', action='store_true',
                        help='Force GPU usage and fail if not available')
    parser.add_argument('--suppress_warnings', action='store_true',
                        help='Suppress all consistency warnings')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed probabilities for each image')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
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

    # Initialize the orientation detection model
    try:
        od = OrientationDetection(
            checkpoint_path=str(model_path),
            device=args.device,
            suppress_warnings=args.suppress_warnings,
            force_gpu=args.force_gpu,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("\nTroubleshooting GPU issues:")
        print("1. Make sure you have PyTorch with CUDA support installed.")
        print("   Run: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118")
        print("2. Check if CUDA toolkit is installed correctly on your system.")
        print("3. Verify NVIDIA drivers are up to date.")
        print("4. Try running with --device cpu to use CPU as fallback.")
        sys.exit(1)

    # Supported image extensions
    # Getting PIL's supported extensions
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

    # Setup CSV output if requested
    csv_writer = None
    csv_file = None
    if args.csv:
        try:
            csv_file = open(args.csv, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            # Write header
            csv_writer.writerow([
                'Image', 'Best Angle', 'Confidence',
                'Prob 0°', 'Prob 90°', 'Prob 180°', 'Prob 270°',
                'Corrected'
            ])
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            args.csv = None

    # Process each image
    corrected_count = 0
    for img_path in image_files:
        try:
            # Read the image
            img = od.read_image(str(img_path))

            # Get the best angle, confidence, and all probabilities
            best_angle, confidence, all_probs = od.get_best_angle(img)

            # Print the result
            if args.verbose:
                print(f"\n{img_path}:")
                for angle, prob in sorted(all_probs.items()):
                    star = "*" if angle == best_angle else " "
                    print(f"{star} {angle}°: {prob * 100:.2f}%")
            else:
                print(f"{img_path}: Orientation {best_angle:03d}° (Confidence: {confidence * 100:.2f}%)")

            corrected = False

            # Correct the orientation if requested and confidence is high enough
            if args.correct and confidence > args.threshold and best_angle != 0:
                # Determine rotation method
                rotation_methods = {
                    90: Image.Transpose.ROTATE_270,  # Counterclockwise to correct 90° clockwise
                    180: Image.Transpose.ROTATE_180,
                    270: Image.Transpose.ROTATE_90,  # Clockwise to correct 270° clockwise
                }

                # Rotate the image
                corrected_img = img.transpose(rotation_methods[best_angle])

                # Save the corrected image
                if output_folder:
                    # Preserve the directory structure relative to input_folder
                    rel_path = img_path.relative_to(input_folder)
                    out_path = output_folder / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_path = img_path

                corrected_img.save(out_path)
                corrected_count += 1
                corrected = True
                print(f"  - Corrected to 0° and saved to {out_path}")

            # Write to CSV if requested
            if csv_writer:
                csv_writer.writerow([
                    str(img_path),
                    best_angle,
                    f"{confidence * 100:.2f}%",
                    f"{all_probs[0] * 100:.2f}%",
                    f"{all_probs[90] * 100:.2f}%",
                    f"{all_probs[180] * 100:.2f}%",
                    f"{all_probs[270] * 100:.2f}%",
                    "Yes" if corrected else "No"
                ])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            if csv_writer:
                csv_writer.writerow([str(img_path), "ERROR", str(e), "", "", "", "", "No"])

    if args.correct:
        print(f"Corrected {corrected_count} out of {len(image_files)} images")

    # Close CSV file if opened
    if csv_file:
        csv_file.close()
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()