import argparse
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def parse_arguments():
    """
    @brief Parses command-line arguments.
    @return Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Enhance CARLA dataset images using a trained TensorRT model.")
    parser.add_argument('--model_trt', action='store', help='The path where the tensorrt engine file (trained model) is stored.')
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
    if (args.model_trt is None) or not os.path.isfile(args.model_trt):
        print('--model_trt argument is not set. Please provide a valid path of the trained TensorRT engine.')
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

def load_trt_engine(engine_path):
    """
    @brief Loads a TensorRT engine from a file.
    @param engine_path Path to the TensorRT engine file.
    @return The loaded TensorRT engine.
    """
    print(f"Loading TensorRT engine from {engine_path}...")
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("Failed to load TensorRT engine.")
        exit(1)
    
    print("TensorRT engine loaded successfully.")
    return engine

def initialize_trt_context(engine):
    """
    @brief Initializes the TensorRT execution context.
    @param engine TensorRT engine.
    @return The initialized TensorRT execution context.
    """
    context = engine.create_execution_context()
    if context is None:
        print("Failed to create execution context.")
        exit(1)
    
    return context

def get_engine_info(engine):
    """
    @brief Extracts input and output tensor information from the TensorRT engine.
    @param engine TensorRT engine.
    @return A dictionary containing tensor information.
    """
    engine_info = {}

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = tuple(engine.get_tensor_shape(tensor_name))
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        tensor_size = int(np.prod([dim if dim > 0 else 1 for dim in tensor_shape]))

        engine_info[tensor_name] = {
            'index': i,
            'shape': tensor_shape,
            'dtype': tensor_dtype,
            'is_input': is_input,
            'size': tensor_size
        }

    return engine_info


def allocate_buffers(engine):
    """
    @brief Allocates GPU and host buffers for TensorRT inference.
    @param engine TensorRT engine.
    @return Dictionaries containing GPU buffers, host buffers, and engine info.
    """
    engine_info = get_engine_info(engine)
    gpu_buffers = {}
    host_buffers = {}

    for tensor_name, info in engine_info.items():
        np_dtype = np.dtype(trt.nptype(info['dtype']))
        buffer_size = info['size']

        gpu_buffer = cuda.mem_alloc(buffer_size * np_dtype.itemsize)
        gpu_buffers[tensor_name] = gpu_buffer

        host_buffer = cuda.pagelocked_empty(buffer_size, dtype=np_dtype)
        host_buffers[tensor_name] = host_buffer

    return gpu_buffers, host_buffers, engine_info

def load_and_preprocess_data(frame_path, gbuffers_path, label_maps_path):
    """
    @brief Loads image and NumPy arrays and preprocesses them for TensorRT inference.
    @param frame_path Path to the input image frame.
    @param gbuffers_path Path to the GBuffers NumPy file.
    @param label_maps_path Path to the Semantic Segmentation label map NumPy file.
    @return A tuple of preprocessed numpy arrays for image, gbuffers, and label map.
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
    
    return img, gbuffers, label_map

def run_inference_trt(context, engine, gpu_buffers, host_buffers, engine_info, 
                      img, gbuffers, label_map):
    """
    @brief Runs TensorRT inference with the given input data.
    @param context TensorRT execution context.
    @param engine TensorRT engine.
    @param gpu_buffers Dictionary of GPU buffers.
    @param host_buffers Dictionary of host buffers.
    @param engine_info Dictionary containing engine binding information.
    @param img Preprocessed image tensor (numpy array).
    @param gbuffers Preprocessed GBuffers tensor (numpy array).
    @param label_map Preprocessed label map tensor (numpy array).
    @return A PyTorch tensor containing the output of the inference.
    """
    infer_timer = time.time()

    stream = cuda.Stream()

    input_tensors = {
        'input': img.astype(np.float32).reshape(-1),
        'gbuffers': gbuffers.astype(np.float32).reshape(-1),
        'onnx::Gather_2': label_map.astype(np.float32).reshape(-1),
    }

    for tensor_name, values in input_tensors.items():
        if tensor_name not in gpu_buffers:
            raise KeyError(f'Missing input tensor buffer for {tensor_name}')
        cuda.memcpy_htod(gpu_buffers[tensor_name], values)

    for tensor_name, gpu_buffer in gpu_buffers.items():
        if tensor_name in input_tensors:
            continue
        context.set_tensor_address(tensor_name, int(gpu_buffer))

    for tensor_name, values in input_tensors.items():
        context.set_tensor_address(tensor_name, int(gpu_buffers[tensor_name]))

    if not context.execute_async_v3(stream_handle=stream.handle):
        raise RuntimeError('TensorRT execution failed.')

    output_tensor_name = None
    for tensor_name, info in engine_info.items():
        if not info['is_input']:
            output_tensor_name = tensor_name
            break

    if output_tensor_name is None:
        print("Could not find output tensor.")
        exit(1)

    cuda.memcpy_dtoh(host_buffers[output_tensor_name], gpu_buffers[output_tensor_name])
    stream.synchronize()

    print("Inference time: " + str(time.time() - infer_timer))

    output_shape = engine_info[output_tensor_name]['shape']
    output_numpy = host_buffers[output_tensor_name].reshape(output_shape)

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

def process_single_frame(context, engine, gpu_buffers, host_buffers, engine_info,
                        frame_path, gbuffers_path, label_maps_path, out_path, file_name):
    """
    @brief Coordinates the preprocessing, inference, and postprocessing for a single frame.
    @param context TensorRT execution context.
    @param engine TensorRT engine.
    @param gpu_buffers Dictionary of GPU buffers.
    @param host_buffers Dictionary of host buffers.
    @param engine_info Dictionary containing engine binding information.
    @param frame_path Path to the input image frame.
    @param gbuffers_path Path to the GBuffers NumPy file.
    @param label_maps_path Path to the Semantic Segmentation label map NumPy file.
    @param out_path Path to the directory where the enhanced image will be saved.
    @param file_name The name of the file to save.
    """
    img, gbuffers, label_map = load_and_preprocess_data(frame_path, gbuffers_path, label_maps_path)
    output_tensor = run_inference_trt(context, engine, gpu_buffers, host_buffers, 
                                     engine_info, img, gbuffers, label_map)
    postprocess_and_save_image(output_tensor, out_path, file_name)

def process_dataset(context, engine, gpu_buffers, host_buffers, engine_info,
                   frames_dir, gbuffers_dir, semseg_dir, out_path):
    """
    @brief Iterates over the dataset frames and processes each one.
    @param context TensorRT execution context.
    @param engine TensorRT engine.
    @param gpu_buffers Dictionary of GPU buffers.
    @param host_buffers Dictionary of host buffers.
    @param engine_info Dictionary containing engine binding information.
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
            context=context,
            engine=engine,
            gpu_buffers=gpu_buffers,
            host_buffers=host_buffers,
            engine_info=engine_info,
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
    
    # Load TensorRT engine
    engine = load_trt_engine(args.model_trt)
    
    # Create execution context
    context = initialize_trt_context(engine)
    
    # Allocate GPU and host buffers
    gpu_buffers, host_buffers, engine_info = allocate_buffers(engine)
    
    # Process dataset
    process_dataset(
        context=context,
        engine=engine,
        gpu_buffers=gpu_buffers,
        host_buffers=host_buffers,
        engine_info=engine_info,
        frames_dir=frames_dir,
        gbuffers_dir=gbuffers_dir,
        semseg_dir=semseg_dir,
        out_path=out_path
    )
    
    # Cleanup
    for buffer in gpu_buffers.values():
        buffer.free()
    
    print("Inference complete. Enhanced images saved to:", out_path)
