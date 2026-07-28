@echo off
setlocal

set "Map_Name=Town10HD_Opt"
set "Weather_List=ClearNoon CloudyNoon WetNoon WetCloudyNoon SoftRainNoon MidRainyNoon HardRainNoon ClearSunset CloudySunset WetSunset WetCloudySunset SoftRainSunset MidRainSunset HardRainSunset"

cd /d "%~dp0"

set "PYTHON_CMD=python"
set "PYTHON_ARGS="

if defined CONDA_PREFIX (
    set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
) else (
    where conda >nul 2>nul
    if not errorlevel 1 (
        conda env list | findstr /I "carla" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_CMD=conda"
            set "PYTHON_ARGS=run -n carla python"
        )
    )

    if /I "%PYTHON_CMD%"=="python" (
        where python >nul 2>nul
        if errorlevel 1 (
            where py >nul 2>nul
            if not errorlevel 1 (
                set "PYTHON_CMD=py"
            )
        )
    )
)

"%PYTHON_CMD%" %PYTHON_ARGS% -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('tensorrt') else 1)" >nul 2>nul
if not errorlevel 1 (
    set "RUN_TRT=1"
) else (
    set "RUN_TRT=0"
)

for %%W in (%Weather_List%) do (
    echo Generating dataset for weather: %%W

    "%PYTHON_CMD%" %PYTHON_ARGS% carla_unreal_engine_5\carla_epe_ue5.py --output_dir .\Dataset\ --map_name %Map_Name% --weather_preset %%W --num_frames_export 300

    "%PYTHON_CMD%" %PYTHON_ARGS% carla_unreal_engine_5\epe_preprocess.py --input_path "%CD%\Dataset\%Map_Name%_%%W" --output_path "%CD%\Dataset\%Map_Name%_%%W" --gbuffers "['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']" --gbuffers_grayscale "['SceneDepth','Metallic','Specular','Roughness']"

    if exist ".\Dataset\models\carla2cityscapes-360000.onnx" (
        "%PYTHON_CMD%" %PYTHON_ARGS% carla_unreal_engine_5\infere_onnx.py --model_onnx .\Dataset\models\carla2cityscapes-360000.onnx --dataset_directory "%CD%\Dataset\%Map_Name%_%%W" --out_path "%CD%\Dataset\%Map_Name%_%%W\Carla2CityScapes"
    ) else (
        echo Skipping ONNX inference for Carla2CityScapes: model not found.
    )

    if exist ".\Dataset\models\carla2kitti-400000.onnx" (
        "%PYTHON_CMD%" %PYTHON_ARGS% carla_unreal_engine_5\infere_onnx.py --model_onnx .\Dataset\models\carla2kitti-400000.onnx --dataset_directory "%CD%\Dataset\%Map_Name%_%%W" --out_path "%CD%\Dataset\%Map_Name%_%%W\Carla2Kitti"
    ) else (
        echo Skipping ONNX inference for Carla2Kitti: model not found.
    )

    if "%RUN_TRT%"=="1" (
        if exist ".\Dataset\models\carla2cityscapes-360000.trt" (
            "%PYTHON_CMD%" %PYTHON_ARGS% carla_unreal_engine_5\infere_trt.py --model_trt .\Dataset\models\carla2cityscapes-360000.trt --dataset_directory "%CD%\Dataset\%Map_Name%_%%W\" --out_path "%CD%\Dataset\%Map_Name%_%%W\Carla2CityScapes"
        ) else (
            echo Skipping TensorRT inference for Carla2CityScapes: model not found.
        )

        if exist ".\Dataset\models\carla2kitti-400000.trt" (
            "%PYTHON_CMD%" %PYTHON_ARGS% carla_unreal_engine_5\infere_trt.py --model_trt .\Dataset\models\carla2kitti-400000.trt --dataset_directory "%CD%\Dataset\%Map_Name%_%%W\" --out_path "%CD%\Dataset\%Map_Name%_%%W\Carla2Kitti"
        ) else (
            echo Skipping TensorRT inference for Carla2Kitti: model not found.
        )
    ) else (
        echo TensorRT is not available in the current Python environment; skipping TensorRT inference.
    )

    timeout /t 5 /nobreak >nul
)

endlocal