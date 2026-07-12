# CARLA2Real-ADAS

You **must** build CARLA from source and run it through the Unreal Engine editor. The packaged version of CARLA locks the Unreal console, which is needed for this setup.

Building from source is simpler than before, and you can find detailed steps in the official [documentation](https://carla-ue5.readthedocs.io/en/latest/) and the [GitHub repository](https://github.com/carla-simulator/carla).

After building CARLA from source, run the following command to run the Unreal editor in game mode (Make sure you have a graphical connection to the computer as this command will open a window):
```bash
~/workset/UnrealEngine5_carla/Engine/Binaries/Linux/UnrealEditor ~/workset/CarlaUE5/Unreal/CarlaUnreal/CarlaUnreal.uproject -game
```

Open another terminal into the project, and make sure that you go into the project and activate the miniconda `carla` environment. Execute the following command to start the data generation procedure:

```bash
conda activate carla
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
python3 carla_epe_ue5.py
# Optional arguments and their corresponding default values:
    # --width 960 --height 540 --output_dir ../Dataset 
    # --num_frames_export 5 --export_step 60 
    # --num_vehicles 100 --num_walkers 40
    # --bbox_distance_range [0.1, 50.0]
    # --map_name Town10HD_Opt --weather_preset ClearNoon
```

> ⚠️ **Warning**: The script will automatically input the high-res command and synchronize it with other data that can be exported via the CARLA Python API. **After executing the script, simply focus the CARLA simulator window with your mouse to proceed.**

Once the data generation is complete, Open a new terminal into the project, and run the following command to convert the collected data into NPZ files of both [GBuffers](#GBuffers-npz-files) and [Semantic Segmentations](#semantic-segmentation-npz-files). Those will be used by the photorealism enhancer:

```bash
conda activate carla
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
python3 epe_preprocess.py --input_path $PWD/../Dataset/ --output_path $PWD/../Dataset/ --gbuffers "['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']" --gbuffers_grayscale "['SceneDepth','Metallic','Specular','Roughness']"
```

The final step is to enhance the data by running inference with the model. You may use any of [CARLA2Real models](https://drive.google.com/drive/folders/1WF1RCE-AUWFXdZdWUt3wMbrbBCHrTVCX) (The scripts supports both ONNX and TRT models. Also we've uploaded the models as part of our kaggle dataset). Then open a new terminal into the project and run the following command to generate the enhanced images out of the collected data:

```bash
conda activate carla
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
mkdir ../Dataset/EnhancedPhotos
python3 infere_onnx.py --model_onnx ../Dataset/models/carla2cityscapes-360000.onnx --dataset_directory $PWD/../Dataset/CarlaUE5-EPE --out_path $PWD/../Dataset/EnhancedPhotos
# OR
python3 infere_trt.py --model_trt ../Dataset/models/carla2cityscapes-360000.trt --dataset_directory $PWD/../Dataset/CarlaUE5-EPE --out_path $PWD/../Dataset/EnhancedPhotos
```

You can then visualize the collected data on top of the enhanced images. Open a new terminal into the project and run the following command:

```bash
conda activate carla
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
python3 visualize_bboxes.py --frames_dir $PWD/../Dataset/EnhancedPhotos --bboxes_dir $PWD/../Dataset/BoundingBoxes/ --output_dir $PWD/../Dataset/VisulOutput/
```
The script saves the images into the output folder that contains the bounding boxes on top of the enhanced images and titled with object type, showing the object distances as well.

---

#### GBuffers npz files:
The total number of channels in the NPZ file will be: 3 + 1 + 3 + 1 + 1 + 1 + 3 + 3 + 1 + 1 = 18 channels
- SceneColor - RGB color information
- SceneDepth - Depth map (grayscale, single channel)
- WorldNormal - World space normals (RGB)
- Metallic - Metallic property (grayscale, single channel)
- Specular - Specular property (grayscale, single channel)
- Roughness - Surface roughness (grayscale, single channel)
- BaseColor - Base color of materials (RGB)
- SubsurfaceColor - Subsurface scattering color (RGB)
- SSAO - Screen Space Ambient Occlusion (random noise, single channel)
- Semantic ID - Raw semantic segmentation class ID (single channel)

> The grayscale buffers (Depth, Metallic, Specular, Roughness) are expanded to have a single channel dimension for consistency.


#### Semantic Segmentation npz files:
One-hot encoded semantic segmentation labels
A single numpy array called label_map with shape (height, width, 12) - containing 12 channels of binary semantic segmentation masks.

The 12 semantic classes (in order):

0. Sky
1. Road (combination of multiple road-related classes)
2. Vehicle (combination of multiple vehicle types)
3. Terrain
4. Vegetation
5. Person (combination of pedestrian types)
6. Infrastructure
7. Traffic Light
8. Traffic Sign
9. Ego (ego vehicle, combination of types)
10. Building (combination of building types)
11. Unlabeled (combination of unlabeled categories)
