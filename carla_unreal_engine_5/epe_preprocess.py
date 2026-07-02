import argparse
import ast
import os
import sys

import cv2
import numpy as np


def split_gt_label(gt_labels, multi_gt_labels):
    """!
    @brief Splits the ground truth semantic map into distinct channel masks.

    @param gt_labels A numpy array representing the semantic map image.
    @param multi_gt_labels A pre-initialized tensor with specific classes for broadcasting.

    @return A concatenated numpy array where each channel represents a different class.
    """
    r = (multi_gt_labels == gt_labels[:, :, 2][:, :, np.newaxis].astype(np.float32))

    class_sky = r[:, :, 0][:, :, np.newaxis]
    class_road = np.any(r[:, :, [1, 2, 3, 4, 5]], axis=2)[:, :, np.newaxis]
    class_vehicle = np.any(r[:, :, [6, 7, 8, 9, 10, 11]], axis=2)[:, :, np.newaxis]
    class_terrain = r[:, :, 12][:, :, np.newaxis]
    class_vegetation = r[:, :, 13][:, :, np.newaxis]
    class_person = np.any(r[:, :, [14, 15]], axis=2)[:, :, np.newaxis]
    class_infa = r[:, :, 16][:, :, np.newaxis]
    class_traffic_light = r[:, :, 17][:, :, np.newaxis]
    class_traffic_sign = r[:, :, 18][:, :, np.newaxis]
    class_ego = np.any(r[:, :, [19, 20]], axis=2)[:, :, np.newaxis]
    class_building = np.any(r[:, :, [21, 22, 23, 24, 25, 26]], axis=2)[:, :, np.newaxis]
    class_unlabeled = np.any(r[:, :, [27, 28]], axis=2)[:, :, np.newaxis]

    concatenated_array = np.concatenate((
        class_sky, class_road, class_vehicle, class_terrain, class_vegetation,
        class_person, class_infa, class_traffic_light, class_traffic_sign, class_ego,
        class_building, class_unlabeled
    ), axis=2)

    return concatenated_array


def initialize_gt_labels(width=960, height=540, num_channels=29):
    """!
    @brief Initializes the multi-channel ground truth labels tensor used for masking.

    @param width Image width.
    @param height Image height.
    @param num_channels Number of channels corresponding to specific classes.

    @return A transposed numpy array initialized with specific label classes.
    """
    specific_classes = [11, 1, 2, 24, 25, 27, 14, 15, 16, 17, 18, 19, 10, 9, 12, 13, 6, 7, 8, 21, 23, 20, 3, 4, 5, 26, 28, 0, 22]
    multi_channel_array = np.zeros((num_channels, height, width))
    for channel_index, value in enumerate(specific_classes):
        multi_channel_array[channel_index, :, :] = value
    
    return np.transpose(multi_channel_array, axes=(1, 2, 0))


def create_out_dir(path):
    """!
    @brief Creates output directories for frames, G-buffers, and semantic segmentation labels.

    @param path The base output directory path.

    @return A tuple containing paths to the frames, G-buffers, and labels directories.
    """
    data_path = os.path.abspath(os.path.join(path, "CarlaUE5-EPE"))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    fpath = os.path.join(data_path, "Frames")
    gbuffpath = os.path.join(data_path, "GBuffers")
    labelspath = os.path.join(data_path, "SemanticSegmentation")

    for dir_path in [fpath, gbuffpath, labelspath]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    return fpath, gbuffpath, labelspath


def parse_args():
    """!
    @brief Parses command line arguments.

    @return Parsed arguments object containing input/output paths and G-buffer configurations.
    """
    parser = argparse.ArgumentParser(description="Process Carla UE5 dataset for EPE.")

    parser.add_argument("--input_path", type=str, default="A:/UE5Dataset", help="Path to the input directory")
    parser.add_argument("--output_path", type=str, default="A:/", help="Path to the output directory")
    parser.add_argument("--gbuffers", type=str, default="['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']", help="The GBuffers names that define the images that will be used to construct the GBuffer matrix.")
    parser.add_argument("--gbuffers_grayscale", type=str, default="['SceneDepth','Metallic','Specular','Roughness']", help="The GBuffers names that define images that are grayscale (Depth,Metallic,etc.).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        print("Error: Input path does not exist.")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args


def load_and_process_gbuffers(frames_path, image_idx, buffers, grayscale_gbuffers, height, width, gt_map):
    """!
    @brief Loads all required G-buffer images and concatenates them.

    @param frames_path Path to the directory containing frame images.
    @param image_idx Current index of the image being processed.
    @param buffers List of G-buffer names to process.
    @param grayscale_gbuffers List of G-buffer names that should be treated as grayscale.
    @param height Image height.
    @param width Image width.
    @param gt_map Ground truth semantic map.

    @return A concatenated numpy array of all processed G-buffers including SSAO and labels.
    """
    gbuff = []
    for buffer_name in buffers:
        buffer_image_path = os.path.join(frames_path, f"{image_idx}_{buffer_name}.png")
        buffer_image = cv2.imread(buffer_image_path)

        if buffer_image is None:
            raise FileNotFoundError(f"Missing G-buffer image: {buffer_image_path}")

        if buffer_name in grayscale_gbuffers:
            buffer_image = buffer_image[:, :, 0]
            buffer_image = np.expand_dims(buffer_image, axis=-1)

        gbuff.append(buffer_image)

    # Adding random SSAO mask and original label semantic channel
    ssao = np.random.rand(height, width, 1)
    gbuff.append(ssao)
    gbuff.append(gt_map[:, :, 2][:, :, np.newaxis])

    return np.concatenate(gbuff, axis=2)


def process_single_frame(image_idx, frames_path, gt_map_path, buffers, grayscale_gbuffers, 
                         out_frame_path, out_gbuffer_path, out_label_path, multi_gt_labels):
    """!
    @brief Processes a single frame and its corresponding G-buffers and semantic map.

    @param image_idx The index of the frame to process.
    @param frames_path Path to the frames directory.
    @param gt_map_path Path to the semantic masks directory.
    @param buffers List of G-buffer names to read.
    @param grayscale_gbuffers List of G-buffer names that are grayscale.
    @param out_frame_path Path to save the final color image.
    @param out_gbuffer_path Path to save the processed G-buffers.
    @param out_label_path Path to save the processed semantic segmentation maps.
    @param multi_gt_labels Pre-initialized array for splitting ground truth labels.

    @return The initialized multi_gt_labels in case it was created during processing, and a boolean indicating success.
    """
    image_path = os.path.join(frames_path, f"{image_idx}.png")
    image = cv2.imread(image_path)

    if image is None:
        return multi_gt_labels, False

    height, width = image.shape[:2]

    if multi_gt_labels is None or multi_gt_labels.size == 0:
        multi_gt_labels = initialize_gt_labels(width=width, height=height, num_channels=29)

    gt_map_file = os.path.join(gt_map_path, f"segmentation_{image_idx}.png")
    gt_map = cv2.imread(gt_map_file)

    if gt_map is None:
        raise FileNotFoundError(f"Missing GT map: {gt_map_file}")

    gbuffers = load_and_process_gbuffers(
        frames_path, image_idx, buffers, grayscale_gbuffers, height, width, gt_map
    )

    label_map = split_gt_label(gt_map, multi_gt_labels)

    print(f"Processed Image: {np.array(image).shape}")
    print(f"Processed Gbuffers: {gbuffers.shape}")
    print(f"Processed Masks: {label_map.shape}")

    cv2.imwrite(os.path.join(out_frame_path, f"FinalColor-{image_idx}.png"), np.array(image))
    np.savez_compressed(os.path.join(out_gbuffer_path, f"GBuffer-{image_idx}.npz"), gbuffers)
    np.savez_compressed(os.path.join(out_label_path, f"SemanticSegmentation-{image_idx}.npz"), label_map)

    return multi_gt_labels, True


if __name__ == "__main__":
    """!
    @brief Main function to execute the Carla UE5 dataset preprocessing pipeline.
    """
    args = parse_args()

    out_frame_path, out_gbuffer_path, out_label_path = create_out_dir(args.output_path)

    buffers = ast.literal_eval(args.gbuffers)
    grayscale_gbuffers = ast.literal_eval(args.gbuffers_grayscale)

    frames_path = os.path.abspath(os.path.join(args.input_path, "Frames"))
    gt_map_path = os.path.abspath(os.path.join(args.input_path, "Semantic"))

    image_idx = 0
    multi_gt_labels = None

    while True:
        image_idx += 1
        try:
            multi_gt_labels, success = process_single_frame(
                image_idx,
                frames_path,
                gt_map_path,
                buffers,
                grayscale_gbuffers,
                out_frame_path,
                out_gbuffer_path,
                out_label_path,
                multi_gt_labels
            )

            if not success:
                # End of images
                break

        except Exception as e:
            print(f"Error processing frame {image_idx}: {e}")
            break

