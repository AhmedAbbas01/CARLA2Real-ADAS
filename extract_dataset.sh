#!/bin/bash

Map_Name=Town10HD_Opt
Weather_List=(ClearNoon CloudyNoon WetNoon WetCloudyNoon SoftRainNoon MidRainyNoon HardRainNoon ClearSunset CloudySunset WetSunset WetCloudySunset SoftRainSunset MidRainSunset HardRainSunset)

for weather in "${Weather_List[@]}"; do
 
 echo "Generating dataset for weather: $weather"
 
 python3 carla_unreal_engine_5/carla_epe_ue5.py --output_dir ./Dataset/ --map_name $Map_Name --weather_preset $weather --num_frames_export 300
 
 python3 carla_unreal_engine_5/epe_preprocess.py --input_path "$PWD/Dataset/$Map_Name"_"$weather" --output_path "$PWD/Dataset/$Map_Name"_"$weather" --gbuffers "['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']" --gbuffers_grayscale "['SceneDepth','Metallic','Specular','Roughness']" &
 
 # Either use ONNX or TensorRT for inference, depending on your preference and setup.
 python3 carla_unreal_engine_5/infere_onnx.py --model_onnx ./Dataset/models/carla2cityscapes-360000.onnx --dataset_directory "$PWD/Dataset/$Map_Name"_"$weather" --out_path "$PWD/Dataset/$Map_Name"_"$weather"/Carla2CityScapes
 python3 carla_unreal_engine_5/infere_onnx.py --model_onnx ./Dataset/models/carla2kitti-400000.onnx      --dataset_directory "$PWD/Dataset/$Map_Name"_"$weather" --out_path "$PWD/Dataset/$Map_Name"_"$weather"/Carla2Kitti
 # OR
 python3 carla_unreal_engine_5/infere_trt.py  --model_trt ./Dataset/models/carla2cityscapes-360000.trt --dataset_directory "$PWD/Dataset/$Map_Name"_"$weather"/  --out_path "$PWD/Dataset/$Map_Name"_"$weather"/Carla2CityScapes 
 python3 carla_unreal_engine_5/infere_trt.py  --model_trt ./Dataset/models/carla2kitti-400000.trt      --dataset_directory "$PWD/Dataset/$Map_Name"_"$weather"/  --out_path "$PWD/Dataset/$Map_Name"_"$weather"/Carla2Kitti

 sleep 5

done

