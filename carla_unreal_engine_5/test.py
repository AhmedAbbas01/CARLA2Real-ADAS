import argparse
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import onnxruntime

def parse_arguments():
    """
    @brief Parses command-line arguments.
    @return Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Enhance CARLA dataset images using a trained ONNX model.")
    parser.add_argument('--model_onnx', action='store', help='The path where the onnx file (trained model) is stored.')
    parser.add_argument('--dataset_directory', action='store', help='The path where the synthetic (fake) dataset is stored.')
    parser.add_argument('--out_path', action='store', help='The path where the enhanced images will be saved.')
    return parser.parse_args()

def validate_arguments(args):
    """
    @brief Validates the provided command-line arguments.
    @param args Parsed command-line arguments.
    @return A tuple containing formatted dataset directory and output path.
    """
    if (args.dataset_directory is None) or not os.path.isdir(args.dataset_directory):
        print('--dataset_directory argument is not set. Please provide a valid path in the disk where the dataset is stored.')
        exit(1)

    if (args.out_path is None):
        print('--out_path argument is not set. Please provide a valid path.')
        exit(1)
    elif not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print(f"Created output directory: {args.out_path}")
    if (args.model_onnx is None) or not os.path.isfile(args.model_onnx):
        print('--model_onnx argument is not set. Please provide a valid path of the trained model.')
        exit(1)

    dataset_directory = args.dataset_directory
    out_path = args.out_path

    if not dataset_directory.endswith(('/', '\\')):
        dataset_directory += '/'
    if not out_path.endswith(('/', '\\')):
        out_path += '/'
        
    return dataset_directory, out_path

def check_dataset_structure(dataset_directory):
    """
    @brief Checks if the dataset directory contains the required subdirectories.
    @param dataset_directory Path to the dataset directory.
    @return A tuple of paths to the Frames, GBuffers, and SemanticSegmentation directories.
    """
    subdirectories = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]
    
    if not ('Frames' in subdirectories and 'GBuffers' in subdirectories and 'SemanticSegmentation' in subdirectories):
        print("The dataset should be structured as CarlaDataset directory that contains Frames, GBuffers, and SemanticSegmentation subdirectories.")
        exit(1)
        
    frames_dir = os.path.join(dataset_directory, 'Frames')
    gbuffers_dir = os.path.join(dataset_directory, 'GBuffers')
    semseg_dir = os.path.join(dataset_directory, 'SemanticSegmentation')
    
    return frames_dir, gbuffers_dir, semseg_dir

def clean_onnx_profiles():
    """
    @brief Deletes older ONNX profile files from the current working directory.
    """
    print("Clearing older ONNX profiles...")
    file_list = os.listdir(os.getcwd())
    for filename in file_list:
        if "onnxruntime_profile__" in filename:
            file_path = os.path.join(os.getcwd(), filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def initialize_onnx_session(model_path):
    """
    @brief Initializes the ONNX inference session and creates an IO binding object.
    @param model_path Path to the ONNX model file.
    @return A tuple containing the initialized session and its corresponding IO binding object.
    """
    opts = onnxruntime.SessionOptions()
    opts.enable_profiling = True
    session = onnxruntime.InferenceSession(
        model_path,
        opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider",],
    )
    io_binding = session.io_binding()
    return session, io_binding

def load_and_preprocess_data(frame_path, gbuffers_path, label_maps_path):
    """
    @brief Loads image and NumPy arrays and preprocesses them for ONNX inference.
    @param frame_path Path to the input image frame.
    @param gbuffers_path Path to the GBuffers NumPy file.
    @param label_maps_path Path to the Semantic Segmentation label map NumPy file.
    @return A tuple of preprocessed ONNX OrtValues for image, gbuffers, and label map.
    """
    img = Image.open(frame_path).convert('RGB')
    gbuffers = np.load(gbuffers_path)['arr_0']
    label_map = np.load(label_maps_path)['arr_0']
    
    img = np.array(img)

    img = np.expand_dims(img, axis=0)
    gbuffers = np.expand_dims(gbuffers, axis=0)
    label_map = np.expand_dims(label_map, axis=0)

    img = np.transpose(img, (0, 3, 1, 2)) / 255.0
    gbuffers = np.transpose(gbuffers, (0, 3, 1, 2))
    label_map = np.transpose(label_map, (0, 3, 1, 2))

    img = img.astype(np.float32)
    gbuffers = gbuffers.astype(np.float32)
    label_map = label_map.astype(np.float32)

    img_ort = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
    gbuffers_ort = onnxruntime.OrtValue.ortvalue_from_numpy(gbuffers, 'cuda', 0)
    label_map_ort = onnxruntime.OrtValue.ortvalue_from_numpy(label_map, 'cuda', 0)
    
    return img_ort, gbuffers_ort, label_map_ort

def bind_inputs_and_outputs(io_binding, img, gbuffers, label_map):
    """
    @brief Binds the input and output tensors to the ONNX IO binding object.
    @param io_binding ONNX IO binding object.
    @param img Preprocessed image tensor.
    @param gbuffers Preprocessed GBuffers tensor.
    @param label_map Preprocessed label map tensor.
    """
    tdtype = np.float32
    io_binding.bind_input(name='input', device_type=img.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=img.shape(), buffer_ptr=img.data_ptr())
    io_binding.bind_input(name='gbuffers', device_type=gbuffers.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=gbuffers.shape(), buffer_ptr=gbuffers.data_ptr())
    io_binding.bind_input(name='onnx::Gather_2', device_type=label_map.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=label_map.shape(), buffer_ptr=label_map.data_ptr())
    io_binding.bind_output('output', device_type='cuda', device_id=0, element_type=tdtype)

def run_inference(session, io_binding):
    """
    @brief Runs the ONNX session with the given IO binding and measures inference time.
    @param session ONNX InferenceSession.
    @param io_binding ONNX IO binding object with bound inputs and outputs.
    @return A PyTorch tensor containing the output of the inference.
    """
    infer_timer = time.time()
    session.run_with_iobinding(io_binding)
    print("Inference time: " + str(time.time() - infer_timer))
    
    output_numpy = io_binding.copy_outputs_to_cpu()[0]
    return torch.from_numpy(output_numpy)

def postprocess_and_save_image(output_tensor, out_path, file_name):
    """
    @brief Postprocesses the inference output tensor and saves it as an image.
    @param output_tensor Output tensor from the inference session.
    @param out_path Path to the directory where the image will be saved.
    @param file_name The name of the file to save the image as.
    """
    enhanced_frame = (output_tensor[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(enhanced_frame)
    image.save(out_path + file_name)

def process_single_frame(session, io_binding, frame_path, gbuffers_path, label_maps_path, out_path, file_name):
    """
    @brief Coordinates the preprocessing, inference, and postprocessing for a single frame.
    @param session ONNX InferenceSession.
    @param io_binding ONNX IO binding object.
    @param frame_path Path to the input image frame.
    @param gbuffers_path Path to the GBuffers NumPy file.
    @param label_maps_path Path to the Semantic Segmentation label map NumPy file.
    @param out_path Path to the directory where the enhanced image will be saved.
    @param file_name The name of the file to save.
    """
    img, gbuffers, label_map = load_and_preprocess_data(frame_path, gbuffers_path, label_maps_path)
    bind_inputs_and_outputs(io_binding, img, gbuffers, label_map)
    output_tensor = run_inference(session, io_binding)
    postprocess_and_save_image(output_tensor, out_path, file_name)

def process_dataset(session, io_binding, frames_dir, gbuffers_dir, semseg_dir, out_path):
    """
    @brief Iterates over the dataset frames and processes each one.
    @param session ONNX InferenceSession.
    @param io_binding ONNX IO binding object.
    @param frames_dir Path to the Frames directory.
    @param gbuffers_dir Path to the GBuffers directory.
    @param semseg_dir Path to the SemanticSegmentation directory.
    @param out_path Path where the enhanced images will be saved.
    """
    files = [f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]

    for file in files:
        frame_id = file.split("-")[1].split(".")[0]
        
        if file.startswith("__"):
            gbuffers_file = f"__GBuffer-{frame_id}.npz"
            semseg_file = f"__SemanticSegmentation-{frame_id}.npz"
        else:
            gbuffers_file = f"GBuffer-{frame_id}.npz"
            semseg_file = f"SemanticSegmentation-{frame_id}.npz"
            
        frame_path = os.path.join(frames_dir, file)
        gbuffers_path = os.path.join(gbuffers_dir, gbuffers_file)
        label_maps_path = os.path.join(semseg_dir, semseg_file)
        
        process_single_frame(
            session=session,
            io_binding=io_binding,
            frame_path=frame_path,
            gbuffers_path=gbuffers_path,
            label_maps_path=label_maps_path,
            out_path=out_path,
            file_name=file
        )


if __name__ == "__main__":
    """
    @brief The main execution flow of the script.
    """
    args = parse_arguments()
    dataset_directory, out_path = validate_arguments(args)
    frames_dir, gbuffers_dir, semseg_dir = check_dataset_structure(dataset_directory)
    
    clean_onnx_profiles()
    
    session, io_binding = initialize_onnx_session(args.model_onnx)
    
    process_dataset(
        session=session,
        io_binding=io_binding,
        frames_dir=frames_dir,
        gbuffers_dir=gbuffers_dir,
        semseg_dir=semseg_dir,
        out_path=out_path
    )

