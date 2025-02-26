import os
from pathlib import Path
import time

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import src.utils.generator as generator
import src.utils.lung_extraction_funcs as le


class SegmentationModel:
    """Main class for medical image segmentation."""

    def __init__(
        self,
        model_path,
        data_path,
        segmentation_threshold=0.5,
        output_path=None,
        verbosity=False,
        pat_dict=None,
        json_path=None,
    ):
        self.verbosity = verbosity
        self.model = self._load_model(model_path)
        self.patient_dict = (
            pat_dict if pat_dict else le.parse_dataset(data_path, img_only=False)
        )
        self.output_path = output_path
        self.segmentation_threshold = float(segmentation_threshold)

        self.patient_generator = generator.Patient_data_generator(
            patient_dict=self.patient_dict,
            predict=True,
            batch_size=1,
            image_size=512,
            shuffle=True,
            use_window=True,
            window_params=[1500, -600],
            resample_int_val=True,
            resampling_step=25,
            extract_lungs=True,
            size_eval=False,
            verbosity=verbosity,
            reshape=True,
            img_only=False,
        )

        self.json_path = json_path

    def _load_model(self, model_path):
        """Load U-Net model v7 with pretrained weights for lung segmentation."""
        model_config_path = os.path.join(model_path, "model_v7.json")
        model_weights_path = os.path.join(model_path, "weights_v7.hdf5")

        with open(model_config_path, "r") as json_file:
            model = keras.models.model_from_json(json_file.read())
        model.load_weights(model_weights_path)
        return model

    def _generate_segmentation(self, input_volume, processing_params):
        """Generate 3D segmentation mask using the loaded model."""
        if self.verbosity:
            print(f"Starting segmentation for volume of shape {input_volume.shape}...")
            timer_start = time.time()

        raw_predictions = np.zeros_like(input_volume)

        for slice_idx in range(len(input_volume)):
            slice_prediction = self.model.predict(
                input_volume[slice_idx, ...].reshape(-1, 512, 512, 1)
            ).reshape(512, 512)
            raw_predictions[slice_idx, ...] = 1 * (
                slice_prediction > self.segmentation_threshold
            )

        # Post-process predictions
        processed_mask = le.max_connected_volume_extraction(raw_predictions)
        final_mask = self._reconstruct_volume(processed_mask, processing_params)
        if self.verbosity:
            print(f"Segmentation completed in {time.time() - timer_start:.2f} seconds")

        return final_mask.astype(np.int8), processed_mask

    def _reconstruct_volume(self, processed_mask, processing_params):
        """Reconstruct mask to match original DICOM dimensions and spacing."""
        reconstructed_volume = np.zeros(
            processing_params["normalized_shape"], dtype=np.uint8
        )

        z_start, z_end = processing_params["z_st"], processing_params["z_end"]
        xy_start, xy_end = processing_params["xy_st"], processing_params["xy_end"]

        if processing_params["crop_type"]:
            reconstructed_volume[z_start:z_end] = processed_mask[
                :, xy_start:xy_end, xy_start:xy_end
            ]
        else:
            reconstructed_volume[z_start:z_end, xy_start:xy_end, xy_start:xy_end] = (
                processed_mask
            )

        if reconstructed_volume.shape != processing_params["original_shape"]:
            return le.resize_3d_img(
                reconstructed_volume,
                processing_params["original_shape"],
                interp=cv2.INTER_NEAREST,
            )
        return reconstructed_volume

    def _get_all_slices(self, volume, segmentation_mask, processing_params):
        """Identify all available slices."""
        all_axial = np.arange(segmentation_mask.shape[0])
        all_coronal = np.arange(segmentation_mask.shape[1])
        all_sagittal = np.arange(segmentation_mask.shape[2])

        if "z_st" in processing_params:
            all_axial -= processing_params["z_st"]
            all_axial = all_axial[(all_axial >= 0) & (all_axial < volume.shape[0])]

        return all_axial, all_coronal, all_sagittal

    def _generate_segmentation_figures(
        self, volume, slice_indices: list, output_dir, processed_mask
    ):
        """Generate prediction figures for all slices."""
        planes = [
            ("axial", slice_indices[0]),
            ("sagittal", slice_indices[1]),
            ("coronal", slice_indices[2]),
        ]

        for plane, indices in tqdm(planes, desc="Processing planes"):
            for idx in tqdm(
                indices,
                desc=f"Generating {plane.capitalize()} Segmented Images",
                position=0,
                leave=True,
            ):
                if plane == "axial":
                    ct_slice = volume[idx, :, :]
                    mask_slice = processed_mask[idx, :, :]
                elif plane == "sagittal":
                    ct_slice = np.flipud(cv2.resize(volume[:, :, idx], (512, 512)))
                    mask_slice = (
                        cv2.resize(
                            np.flipud(processed_mask[:, :, idx]).astype(np.float32),
                            (512, 512),
                        )
                        > 0.3
                    )
                else:  # coronal
                    ct_slice = np.flipud(cv2.resize(volume[:, idx, :], (512, 512)))
                    mask_slice = (
                        cv2.resize(
                            np.flipud(processed_mask[:, idx, :]).astype(np.float32),
                            (512, 512),
                        )
                        > 0.3
                    )

                if np.sum(mask_slice) == 0:
                    continue

                plt.figure(figsize=(5, 5))
                plt.imshow(ct_slice, cmap="bone")
                plt.title(f"Original {plane.capitalize()} Slice")
                plt.axis("off")
                orig_path = os.path.join(
                    output_dir, f"original_slice_{plane}_{idx}.png"
                )
                plt.savefig(orig_path, bbox_inches="tight", dpi=100, pad_inches=0)
                plt.close()

                masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
                plt.figure(figsize=(5, 5))
                plt.imshow(ct_slice, cmap="bone")
                plt.imshow(
                    masked_mask,
                    cmap="Reds",
                    alpha=0.7,
                    norm=plt.Normalize(vmin=0, vmax=1),
                )
                plt.title(f"Tumor Segmentation - {plane.capitalize()} Slice")
                plt.axis("off")
                seg_path = os.path.join(
                    output_dir, f"segmented_slice_{plane}_{idx}.png"
                )
                plt.savefig(seg_path, bbox_inches="tight", dpi=100, pad_inches=0)
                plt.close()

    def _save_results(
        self,
        segmentation_mask,
        processing_params,
        source_file_path,
        volume_array,
        processed_mask,
    ):
        """Save segmentation results and original image to NRRD format."""

        if isinstance(source_file_path, list):
            source_file_path = source_file_path[0]

        patient_id = Path(source_file_path).parent.name
        output_dir = Path(self.output_path) / f"{patient_id}_(DL)"
        output_dir.mkdir(parents=True, exist_ok=True)

        sitk_mask = sitk.GetImageFromArray(segmentation_mask)
        sitk_mask.SetSpacing(processing_params["original_spacing"])
        sitk_mask.SetOrigin(processing_params["img_origin"])
        sitk.WriteImage(sitk_mask, os.path.join(output_dir, "DL_mask.nrrd"))

        original_image = sitk.ReadImage(source_file_path)
        sitk.WriteImage(original_image, os.path.join(output_dir, "image.nrrd"))

        axial_slices, coronal_slices, sagittal_slices = self._get_all_slices(
            volume_array, segmentation_mask, processing_params
        )

        self._generate_segmentation_figures(
            volume_array,
            [axial_slices, coronal_slices, sagittal_slices],
            output_dir,
            processed_mask=processed_mask,
        )

        return output_dir

    def _calculate_dice_coefficient(self, y_pred, y_true):
        """
        Calculate Dice coefficient between predicted and ground truth masks.

        Args:
            y_pred: Binary prediction mask
            y_true: Binary ground truth mask

        Returns:
            float: Dice coefficient between 0 and 1
        """

        y_pred = np.asarray(y_pred).astype(bool)
        y_true = np.asarray(y_true).astype(bool)

        intersection = np.logical_and(y_pred, y_true).sum()
        total_elements = y_pred.sum() + y_true.sum()

        if total_elements == 0:
            return 1.0

        return (2.0 * intersection) / total_elements

    def _visualize_ground_truth(
        self, volume, ground_truth_mask, processing_params, output_dir
    ):
        """
        Visualize and save ground truth segmentation masks overlaid on original volume.

        Args:
            volume: Original CT volume
            ground_truth_mask: Ground truth segmentation mask
            processing_params: Dictionary containing reconstruction parameters
            output_dir: Directory to save the visualization images
        """
        output_dir = Path(output_dir) / "ground_truth_visualization"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_axial = np.arange(ground_truth_mask.shape[0])

        if "z_st" in processing_params:
            all_axial -= processing_params["z_st"]
            all_axial = all_axial[(all_axial >= 0) & (all_axial < volume.shape[0])]

        for idx in tqdm(
            all_axial,
            desc="Generating Ground Truth Visualizations",
            position=0,
            leave=True,
        ):
            ct_slice = volume[idx, :, :]
            mask_slice = ground_truth_mask[idx, :, :]

            if np.sum(mask_slice) == 0:
                continue

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(ct_slice, cmap="bone")
            plt.title(f"Original Slice {idx}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(ct_slice, cmap="bone")

            masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
            plt.imshow(
                masked_mask,
                cmap="Reds",
                alpha=0.7,
                norm=plt.Normalize(vmin=0, vmax=1),
            )
            plt.title(f"Ground Truth Overlay - Slice {idx}")
            plt.axis("off")

            plt.tight_layout()

            plt.savefig(
                output_dir / f"ground_truth_slice_{idx:03d}.png",
                bbox_inches="tight",
                dpi=150,
                pad_inches=0.1,
            )
            plt.close()

        if self.verbosity:
            print(f"Visualizations saved to: {output_dir}")

    def segment(self):
        """Main pipeline execution method."""

        dice_scores = []

        for volume, ground_truth_mask, file_path, params in tqdm(
            self.patient_generator, desc="Processing patients"
        ):
            volume_array = np.squeeze(volume[0])
            ground_truth = np.squeeze(ground_truth_mask[0])
            processing_params = params[0]

            if self.output_path is not None:
                if isinstance(file_path, list):
                    patient_id = Path(file_path[0]).parent.name
                else:
                    patient_id = Path(file_path).parent.name

                output_dir = Path(self.output_path) / f"{patient_id}_(DL)"

                self._visualize_ground_truth(
                    volume_array, ground_truth, processing_params, output_dir
                )

            segmentation_mask, processed_mask = self._generate_segmentation(
                volume_array, processing_params
            )

            dice_score = self._calculate_dice_coefficient(processed_mask, ground_truth)
            dice_scores.append(dice_score)

            if self.verbosity:
                print(f"Dice coefficient: {dice_score:.4f}")

            if self.output_path is not None:
                _ = self._save_results(
                    segmentation_mask,
                    processing_params,
                    file_path,
                    volume_array,
                    processed_mask,
                )

            with open(self.json_path, "a") as f:
                f.write(f'{{"Dice Coefficient Score": {dice_score:.4f}}}\n')
