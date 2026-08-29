#!/usr/bin/env bash

# This file is sourced by both entry points, so these constants are consumed by callers.
# shellcheck disable=SC2034
SETUP_SCHEMA="imu-benchmark-wsl-environment-v3"
LEGACY_SETUP_SCHEMA="imu-benchmark-wsl-environment-v2"
LEGACY_ENVIRONMENT_SHA256="f1eebb5a3700c6f819210dc17c2b4802629db9ea2629684e5daf1119704ee084"
LEGACY_TORCH_SHA256="0d548c4487efef754acbc89ebf6bf8e35caeda3ad87a985e537b3e3d00058bdb"
MINIFORGE_VERSION="26.3.2-2"
MINIFORGE_SHA256="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
GCLOUD_VERSION="581.0.0"
GCLOUD_SHA256="deffdbe82ca6e3d19ffb291d063a651488e04e1b33799b5a238e4b5c6784e3c6"
GCLOUD_URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-x86_64.tar.gz"
MIN_FREE_KIB=$((25 * 1024 * 1024))

WORK_ROOT="${IMU_BENCH_WORK_ROOT:-${HOME}/imu-fall-work}"

require_absolute_work_root() {
  case "${WORK_ROOT}" in
    /*) ;;
    *)
      echo "ERROR: IMU_BENCH_WORK_ROOT must be an absolute path." >&2
      return 1
      ;;
  esac
}

is_wsl2() {
  grep -Eqi 'microsoft-standard-wsl2' /proc/sys/kernel/osrelease 2>/dev/null
}

require_wsl2() {
  if ! is_wsl2; then
    echo "ERROR: this command is supported only inside WSL2." >&2
    return 1
  fi
  case "${REPO_ROOT}" in
    /mnt/*)
      echo "ERROR: keep the repository in the WSL Linux filesystem, not under /mnt/." >&2
      return 1
      ;;
  esac
}

content_sha256() {
  sha256sum "$1" | cut -d ' ' -f1
}

dependency_digest() {
  {
    printf '%s\n' "${SETUP_SCHEMA}"
    content_sha256 "${REPO_ROOT}/environment.yml"
    content_sha256 "${REPO_ROOT}/requirements-torch-cu129.txt"
    content_sha256 "${REPO_ROOT}/requirements-runtime.txt"
  } | sha256sum | cut -c1-16
}

application_digest() {
  {
    content_sha256 "${REPO_ROOT}/pyproject.toml"
    content_sha256 "${REPO_ROOT}/requirements-runtime.txt"
  } | sha256sum | cut -c1-16
}

legacy_dependency_digest() {
  {
    printf '%s\n' "${LEGACY_SETUP_SCHEMA}"
    printf '%s  %s\n' "${LEGACY_ENVIRONMENT_SHA256}" "${REPO_ROOT}/environment.yml"
    printf '%s  %s\n' "${LEGACY_TORCH_SHA256}" \
      "${REPO_ROOT}/requirements-torch-cu129.txt"
  } | sha256sum | cut -c1-16
}

project_miniforge() {
  printf '%s/toolchains/miniforge-%s/bin/conda\n' "${WORK_ROOT}" "${MINIFORGE_VERSION}"
}

project_gcloud() {
  printf '%s/toolchains/google-cloud-sdk-%s/bin/gcloud\n' "${WORK_ROOT}" "${GCLOUD_VERSION}"
}

find_conda() {
  if [[ -n "${IMU_BENCH_CONDA_EXE:-}" ]]; then
    printf '%s\n' "${IMU_BENCH_CONDA_EXE}"
  else
    project_miniforge
  fi
}

find_gcloud() {
  if [[ -n "${IMU_BENCH_GCLOUD_EXE:-}" ]]; then
    printf '%s\n' "${IMU_BENCH_GCLOUD_EXE}"
  else
    project_gcloud
  fi
}

canonical_environment_path() {
  printf '%s/envs/%s\n' "${WORK_ROOT}" "$(dependency_digest)"
}

legacy_environment_path() {
  printf '%s/envs/%s\n' "${WORK_ROOT}" "$(legacy_dependency_digest)"
}

resolved_environment_path() {
  local candidate
  candidate="$(canonical_environment_path)"
  [[ -e "${candidate}" ]] || return 1
  readlink -f -- "${candidate}"
}

environment_marker() {
  printf '%s/.imu-benchmark-complete-v3\n' "$1"
}

legacy_environment_is_ready() {
  local candidate marker digest
  candidate="$(legacy_environment_path)"
  marker="${candidate}/.imu-benchmark-complete"
  digest="$(legacy_dependency_digest)"
  [[ -d "${candidate}" && -f "${marker}" ]] || return 1
  [[ "$(tr -d '\r\n' <"${marker}")" == "${digest}" ]]
}

legacy_environment_is_compatible() {
  local candidate=$1
  [[ -x "${candidate}/bin/python" ]] || return 1
  "${candidate}/bin/python" -c '
import sys

import cuml
import cupy
import h5py
import numpy
import sklearn
import torch
import xgboost

assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == "2.8.0+cu129"
assert torch.version.cuda == "12.9"
assert xgboost.__version__ == "3.4.0"
assert cuml.__version__ == "26.08.00"
assert cupy.__version__ == "14.2.0"
assert h5py.__version__ == "3.16.0"
assert numpy.__version__ == "2.4.6"
assert sklearn.__version__ == "1.9.0"
' >/dev/null 2>&1
}

find_compatible_legacy_environment() {
  local candidate marker marker_value preferred
  preferred="$(legacy_environment_path)"
  if legacy_environment_is_ready && legacy_environment_is_compatible "${preferred}"; then
    printf '%s\n' "${preferred}"
    return
  fi
  for marker in "${WORK_ROOT}"/envs/*/.imu-benchmark-complete; do
    [[ -f "${marker}" ]] || continue
    candidate="${marker%/.imu-benchmark-complete}"
    marker_value="$(tr -d '\r\n' <"${marker}")"
    [[ "${marker_value}" == "$(basename -- "${candidate}")" ]] || continue
    if legacy_environment_is_compatible "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}
