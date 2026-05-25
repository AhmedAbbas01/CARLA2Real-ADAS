import subprocess
import sys
import yaml

def run_script(script_name):
    print(f"\nRunning {script_name}...\n")

    result = subprocess.run(
        [sys.executable, script_name],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    if result.returncode != 0:
        print(f"\nError: {script_name} failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nFinished {script_name}\n")


if __name__ == "__main__":
    with open("../code/config/carla_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    im2im_model = config["general"]["method"]

    run_script("flux.py")
    
    print(im2im_model)
    if im2im_model == "REGEN":
        run_script("regen.py")
    elif im2im_model == "HYPERGAN":
        run_script("hypergan.py")
    else:
        print("The available im2im translation models are ['REGEN','HYPERGAN'].")
        exit(1)

    print("Pipeline completed successfully.")
