#!/usr/bin/env bash
set -euo pipefail

output=""
seconds=240
display=1
dry_run=0

usage() {
  echo "usage: record_qa_evidence.sh --output /absolute/path.mp4 [--seconds 15-300] [--display N] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output=${2:-}
      shift 2
      ;;
    --seconds)
      seconds=${2:-}
      shift 2
      ;;
    --display)
      display=${2:-}
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$output" || "$output" != /* || "$output" != *.mp4 ]]; then
  echo "--output must be an absolute .mp4 path" >&2
  exit 2
fi
if ! [[ "$seconds" =~ ^[0-9]+$ ]] || (( seconds < 15 || seconds > 300 )); then
  echo "--seconds must be between 15 and 300" >&2
  exit 2
fi
if ! [[ "$display" =~ ^[1-9][0-9]*$ ]]; then
  echo "--display must be a positive integer" >&2
  exit 2
fi
if [[ $(uname -s) != "Darwin" ]]; then
  echo "screen recording requires macOS screencapture" >&2
  exit 3
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 2
fi

temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/geist-qa-recording.XXXXXX")
recording="$temporary_dir/source.mov"
converted="$temporary_dir/converted.m4v"
cleanup() {
  rm -f "$recording" "$converted"
  rmdir "$temporary_dir" 2>/dev/null || true
}
trap cleanup EXIT

if (( dry_run == 1 )); then
  echo "screencapture -v -V${seconds} -D${display} -k <temporary.mov>"
  echo "avconvert --source <temporary.mov> --preset Preset1280x720 --output <temporary.m4v> --replace --disableMetadataFilter"
  echo "final output: $output"
  exit 0
fi

mkdir -p "$(dirname "$output")"
screencapture -v "-V${seconds}" "-D${display}" -k "$recording"
avconvert \
  --source "$recording" \
  --preset Preset1280x720 \
  --output "$converted" \
  --replace \
  --disableMetadataFilter
mv "$converted" "$output"
echo "$output"
