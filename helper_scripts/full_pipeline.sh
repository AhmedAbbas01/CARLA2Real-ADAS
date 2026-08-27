#!/bin/bash

FOLDER_NAME=$PWD/Dataset/Town10HD_Opt_MidRainSunset


python3 carla_unreal_engine_5/epe_preprocess.py --input_path "$FOLDER_NAME" --output_path "$FOLDER_NAME" --gbuffers "['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']" --gbuffers_grayscale "['SceneDepth','Metallic','Specular','Roughness']"

python3 carla_unreal_engine_5/infere_onnx.py --model_onnx ./Dataset/models/carla2cityscapes-360000.onnx --dataset_directory "$FOLDER_NAME/CarlaUE5-EPE" --out_path "$FOLDER_NAME/EnhancedPhotos"

python3 carla_unreal_engine_5/visualize_bboxes.py --frames_dir "$FOLDER_NAME/EnhancedPhotos" --bboxes_dir "$FOLDER_NAME/BoundingBoxes/" --output_dir "$FOLDER_NAME/VisualOutput/"
