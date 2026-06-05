#!/bin/sh

# Displays usage information
usage() {
  echo "Usage: $0 --license LICENSE --python PYTHON INPUT OUTPUT"
  echo "Arguments:"
  echo "  -l, --license LICENSE    Path to FreeSurfer license (required)"
  echo "  -p, --python PYTHON      Path to Python-executable to use (required)"
  echo "  INPUT                    T1 image (.nii.gz) or directory of images (required)"
  echo "  OUTPUT                   Directory where FastSurfer data is written (required)"
}

LICENSE=""
PYTHON=""
INPUT=""
OUTPUT=""

while [ $# -gt 0 ]; do
  case $1 in
    -l|--license)
      LICENSE="$2"
      shift 2
      ;;
    -p|--python)
      PYTHON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      # Handle positional arguments
      if [ -z "$INPUT" ]; then
        INPUT="$1"
      elif [ -z "$OUTPUT" ]; then
        OUTPUT="$1"
      else
        echo "Error: Too many positional arguments"
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

# Validate that exactly two positional arguments are provided
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Error: This script requires exactly two arguments: INPUT and OUTPUT"
  usage
  exit 1
fi

if [ -z "$LICENSE" ]; then
  echo "Error: License is required"
  usage
  exit 1
fi
if [ -z "$PYTHON" ]; then
  echo "Error: Python is required"
  usage
  exit 1
fi

if [ -z "$FASTSURFER_HOME" ]; then
  echo "Error: FASTSURFER_HOME is not set"
  exit 1
fi

if [ ! -e "$INPUT" ]; then
  echo "Error: Input '$INPUT' does not exist"
  exit 1
fi

if [ -f "$INPUT" ]; then
  # Single image: call run_fastsurfer.sh directly
  IMAGE=$(basename "$INPUT" .nii.gz)
  if [ ! -f "$OUTPUT/$IMAGE/mri/mask.mgz" ]; then
    $FASTSURFER_HOME/run_fastsurfer.sh \
      --sd "$OUTPUT" \
      --sid "$IMAGE" \
      --t1 "$INPUT" \
      --fs_license "$LICENSE" \
      --py "$PYTHON" \
      --seg_only \
      --allow_root
  else
    echo "$IMAGE already processed"
  fi
else
  NIFTI_FILES=$(find "$INPUT" -name "*.nii.gz" -type f)
  echo "Found $(echo "$NIFTI_FILES" | wc -l) .nii.gz files"

  # Extract unique filenames and check for duplicates
  UNIQUE_FILENAMES=$(echo "$NIFTI_FILES" | xargs -n1 basename | sort | uniq)
  FILENAME_COUNT=$(echo "$UNIQUE_FILENAMES" | wc -l)
  TOTAL_COUNT=$(echo "$NIFTI_FILES" | wc -l)

  if [ "$FILENAME_COUNT" -ne "$TOTAL_COUNT" ]; then
    echo "Error: Duplicate filenames detected!"
    echo "Duplicate filenames:"
    echo "$NIFTI_FILES" | xargs -n1 basename | sort | uniq -d
    exit 1
  fi

  # Build a FreeSurfer-organized staging directory: <subject>/mri/orig.nii.gz
  # symlinked to the original NIfTI. run_prediction.py --in_dir then processes
  # all subjects in a single Python process (one CUDA context for the whole
  # batch), avoiding the per-subject CUDA teardown/rebuild cycle that triggers
  # PCIe faults on affected driver versions. The staging directory also makes
  # it easy to see how far the run got if the script is interrupted.
  STAGING_DIR=$(mktemp -d /tmp/fastsurfer_staging_XXXXXX)

  echo "$NIFTI_FILES" | while IFS= read -r filepath; do
    IMAGE=$(basename "$filepath" .nii.gz)
    if [ ! -f "$OUTPUT/$IMAGE/mri/mask.mgz" ]; then
      mkdir -p "$STAGING_DIR/$IMAGE/mri"
      ln -s "$(realpath "$filepath")" "$STAGING_DIR/$IMAGE/mri/orig.nii.gz"
    else
      echo "$IMAGE already processed"
    fi
  done

  PENDING=$(find "$STAGING_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

  if [ "$PENDING" -eq 0 ]; then
    echo "All subjects already processed"
    rm -rf "$STAGING_DIR"
    exit 0
  fi

  echo "Processing $PENDING subjects in a single CUDA session"

  export PYTHONPATH="$FASTSURFER_HOME:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=0

  $PYTHON "$FASTSURFER_HOME/FastSurferCNN/run_prediction.py" \
    --in_dir "$STAGING_DIR" \
    --t1 mri/orig.nii.gz \
    --sd "$OUTPUT" \
    --vox_size min \
    --batch_size 1 \
    --viewagg_device auto \
    --device auto \
    --allow_root

  STATUS=$?
  rm -rf "$STAGING_DIR"
  exit $STATUS
fi

