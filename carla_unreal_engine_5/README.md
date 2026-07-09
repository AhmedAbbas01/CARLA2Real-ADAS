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

Once the data generation is complete, Open a new terminal into the project, and run the following command to convert the data into a format compatible with the photorealism enhancement:

```bash
conda activate carla
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
python3 epe_preprocess.py --input_path $PWD/../Dataset/ --output_path $PWD/../Dataset/ --gbuffers "['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']" --gbuffers_grayscale "['SceneDepth','Metallic','Specular','Roughness']"
```

The final step is to enhance the data by running inference with the model. Make sure to download the [ONNX model](https://drive.google.com/drive/folders/1NNGdwxH5anAPbvaM-0Mm3cq8PexjtGtF?usp=drive_link). Then open a new terminal into the project and run the following command to generate the enhanced images out of the collected data:

```bash
conda activate carla
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
mkdir ../Dataset/EnhancedPhotos
python3 test.py --model_onnx ~/Downloads/carla2cityscapes-360000.onnx --dataset_directory $PWD/../Dataset/CarlaUE5-EPE --out_path $PWD/../Dataset/EnhancedPhotos
```

> ⚠️ **Warning**: ONNX Runtime library will require a large amount of VRAM. If you have a limited amount of GPU VRAM then the inference should be performed with the PyTorch code as described in the [CARLA2Real readme](https://github.com/stefanos50/CARLA2Real).

You can then visualize the collected data on top of the enhanced images. Open a new terminal into the project and run the following command:

```bash
conda deactivate && conda deactivate
source /opt/pytorch/bin/activate
cd ~/workset/CARLA2Real-ADAS/carla_unreal_engine_5
python3 visualize_bboxes.py --frames_dir $PWD/../Dataset/EnhancedPhotos --bboxes_dir $PWD/../Dataset/BoundingBoxes/ --output_dir $PWD/../Dataset/VisulOutput/
```
The script saves the images into the output folder that contains the bounding boxes on top of the enhanced images and titled with object type, showing the object distances as well.
