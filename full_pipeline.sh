#!/bin/bash

python3 carla_unreal_engine_5/epe_preprocess.py --input_path $PWD/Dataset/ --output_path $PWD/Dataset/ --gbuffers "['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']" --gbuffers_grayscale "['SceneDepth','Metallic','Specular','Roughness']"

python3 carla_unreal_engine_5/test.py --model_onnx ~/Downloads/carla2cityscapes-360000.onnx --dataset_directory $PWD/Dataset/CarlaUE5-EPE --out_path $PWD/Dataset/EnhancedPhotos 

python3 carla_unreal_engine_5/visualize_bboxes.py --frames_dir $PWD/Dataset/EnhancedPhotos --bboxes_dir $PWD/Dataset/BoundingBoxes/ --output_dir $PWD/Dataset/VisulOutput/


