#!/bin/sh

# Displays usage information
usage() {
  echo "Usage: $0 --license LICENSE --python PYTHON INPUT OUTPUT"
  echo "Arguments:"
  echo "  -l, --license LICENSE    Path to FreeSurfer license (required)"
  echo "  -p, --python PYTHON      Path to Python-executable to use (required)"
  echo "  INPUT                    Input file (required)"
  echo "  OUTPUT                   Output file (required)"
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

if [ ! -d "$INPUT" ]; then
  echo "Error: Input directory '$INPUT' does not exist"
  exit 1
fi

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

# Loop through each NIFTI file path
echo "$NIFTI_FILES" | while IFS= read -r filepath; do
  IMAGE=$(basename "$filepath" .nii.gz)
  if [ ! -f "$OUTPUT/$IMAGE/mri/mask.mgz" ]; then
    $FASTSURFER_HOME/run_fastsurfer.sh \
      --sd $OUTPUT \
      --sid $IMAGE \
      --t1 $filepath \
      --fs_license $LICENSE \
      --py $PYTHON \
      --seg_only
  else
    echo "$IMAGE already processed"
  fi
done

