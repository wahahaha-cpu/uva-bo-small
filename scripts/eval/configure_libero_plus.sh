#!/usr/bin/env bash
set -euo pipefail

# Keep LIBERO-Plus isolated from the original LIBERO package used by UVA.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${PLUS_ROOT:-/data1/local_userdata/jinboning/LIBERO-plus}}"
LIBERO_PLUS_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-${PLUS_CONFIG_PATH:-/data1/local_userdata/jinboning/LIBERO-plus-runtime}}"
PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/vla-adapter-assets/envs/vla-adapter-jepa-gpu/bin/python}"
IMAGEMAGICK_ROOT="${IMAGEMAGICK_ROOT:-/data1/local_userdata/jinboning/vla-adapter-assets/system-libs/imagemagick6}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-0}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"
ASSETS_URL="${ASSETS_URL:-https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip?download=true}"
ASSETS_SIZE="${ASSETS_SIZE:-6395849578}"

if [[ ! -d "$LIBERO_PLUS_ROOT/libero/libero" ]]; then
  echo "LIBERO-Plus source tree is missing: $LIBERO_PLUS_ROOT" >&2
  echo "Clone it with: git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git $LIBERO_PLUS_ROOT" >&2
  exit 2
fi
for toggle_name in INSTALL_RUNTIME_DEPS DOWNLOAD_ASSETS; do
  toggle_value="${!toggle_name}"
  if [[ "$toggle_value" != "0" && "$toggle_value" != "1" ]]; then
    echo "$toggle_name must be 0 or 1; got: $toggle_value" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$LIBERO_PLUS_CONFIG_PATH"
tmp_config="$(mktemp "${TMPDIR:-/tmp}/libero-plus-config.XXXXXX")"
cleanup() {
  rm -f -- "$tmp_config"
  if [[ -n "${package_dir:-}" && -d "${package_dir:-}" ]]; then
    rm -rf -- "$package_dir"
  fi
}
trap cleanup EXIT
sed "s|@LIBERO_PLUS_ROOT@|$LIBERO_PLUS_ROOT|g" \
  "$PROJECT_ROOT/configs/libero_plus.yaml" > "$tmp_config"
install -m 0644 "$tmp_config" "$LIBERO_PLUS_CONFIG_PATH/config.yaml"

if [[ "$INSTALL_RUNTIME_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install wand scikit-image

  package_dir="$(mktemp -d "${TMPDIR:-/tmp}/libero-plus-imagemagick.XXXXXX")"
  apt_packages=(
    imagemagick-6-common
    libmagickcore-6.q16-7t64
    libmagickwand-6.q16-7t64
    liblqr-1-0
    libltdl7
  )
  (cd "$package_dir" && apt-get download "${apt_packages[@]}")
  mkdir -p "$IMAGEMAGICK_ROOT"
  for package_path in "$package_dir"/*.deb; do
    dpkg-deb -x "$package_path" "$IMAGEMAGICK_ROOT"
  done
  magick_lib_dir="$IMAGEMAGICK_ROOT/usr/lib"
  mkdir -p "$magick_lib_dir"
  ln -sfn x86_64-linux-gnu/libMagickWand-6.Q16.so.7 \
    "$magick_lib_dir/libMagickWand-6.Q16.so"
  ln -sfn x86_64-linux-gnu/libMagickCore-6.Q16.so.7 \
    "$magick_lib_dir/libMagickCore-6.Q16.so"
  ln -sfn x86_64-linux-gnu/libMagickWand-6.Q16.so.7 \
    "$magick_lib_dir/libMagickWand-6.Q16.so.6"
  ln -sfn x86_64-linux-gnu/libMagickCore-6.Q16.so.7 \
    "$magick_lib_dir/libMagickCore-6.Q16.so.6"
fi

if [[ "$DOWNLOAD_ASSETS" == "1" ]]; then
  archive_dir="$LIBERO_PLUS_ROOT/.downloads"
  archive_path="$archive_dir/assets.zip"
  mkdir -p "$archive_dir"
  current_size=0
  if [[ -f "$archive_path" ]]; then
    current_size="$(stat -c '%s' "$archive_path")"
  fi
  if (( current_size < ASSETS_SIZE )); then
    echo "Downloading LIBERO-Plus assets to $archive_path (about 6.4 GB)"
    curl -L --fail --retry 5 --retry-delay 5 -C - \
      -o "$archive_path" "$ASSETS_URL"
  else
    echo "Using existing LIBERO-Plus asset archive: $archive_path"
  fi
  final_size="$(stat -c '%s' "$archive_path")"
  if (( final_size < ASSETS_SIZE )); then
    echo "Asset archive is incomplete: $final_size bytes (expected at least $ASSETS_SIZE)" >&2
    exit 2
  fi
  echo "Extracting LIBERO-Plus assets under $LIBERO_PLUS_ROOT/libero/libero"
  unzip -q -o "$archive_path" -d "$LIBERO_PLUS_ROOT/libero/libero"
  assets_target="$LIBERO_PLUS_ROOT/libero/libero/assets"
  assets_source="$(find "$LIBERO_PLUS_ROOT/libero/libero" -type d -path '*/assets' -print -quit)"
  if [[ -n "$assets_source" && "$assets_source" != "$assets_target" ]]; then
    if [[ -e "$assets_target" ]]; then
      echo "Asset target already exists; leaving nested archive copy at $assets_source" >&2
    else
      mv "$assets_source" "$assets_target"
      find "$LIBERO_PLUS_ROOT/libero/libero/inspire" -depth -type d -empty -delete 2>/dev/null || true
    fi
  fi
fi

echo "LIBERO-Plus source: $LIBERO_PLUS_ROOT"
echo "LIBERO-Plus config: $LIBERO_PLUS_CONFIG_PATH/config.yaml"
if "$PYTHON_BIN" -c 'import skimage, wand' >/dev/null 2>&1; then
  echo "Python extras: wand and scikit-image are present"
else
  echo "Python extras are missing. Run INSTALL_RUNTIME_DEPS=1 $0"
fi
if [[ -f "$IMAGEMAGICK_ROOT/usr/lib/x86_64-linux-gnu/libMagickWand-6.Q16.so.7" ]]; then
  echo "ImageMagick runtime: $IMAGEMAGICK_ROOT"
else
  echo "ImageMagick runtime is missing. Run INSTALL_RUNTIME_DEPS=1 $0"
fi
if [[ -d "$LIBERO_PLUS_ROOT/libero/libero/assets" ]]; then
  echo "Asset directory: $LIBERO_PLUS_ROOT/libero/libero/assets"
else
  echo "Asset directory is not present yet. Run DOWNLOAD_ASSETS=1 $0 to fetch and extract assets."
fi
