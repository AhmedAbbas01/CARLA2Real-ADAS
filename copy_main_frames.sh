#!/bin/bash

set -euo pipefail

BASE_DIR=${1:-Dataset}

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: base directory '$BASE_DIR' does not exist."
  exit 1
fi

echo "Scanning dataset base: $BASE_DIR"

found=0
while IFS= read -r -d '' frames_dir; do
  found=1
  target_dir="$frames_dir/../Frames_main"
  mkdir -p "$target_dir"
  echo "Copying main frames from '$frames_dir' to '$target_dir'"

  # Copy only files whose basename is digits followed by .png
  while IFS= read -r -d '' frame_file; do
    cp --update=none "$frame_file" "$target_dir/"
  done < <(find "$frames_dir" -maxdepth 1 -type f -regextype posix-extended -regex '.*/[0-9]+\.png' -print0)

done < <(find "$BASE_DIR" -type d -name Frames -print0)

if [[ "$found" -eq 0 ]]; then
  echo "No Frames directories found under '$BASE_DIR'."
  exit 1
fi

echo "Done."
