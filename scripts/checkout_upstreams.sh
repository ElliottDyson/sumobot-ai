#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="${root_dir}/.upstreams"
mkdir -p "${upstream_dir}"

checkout() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="${upstream_dir}/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

checkout Newton https://github.com/newton-physics/newton.git 82526c0aa7322569de4faf461b0db87b294a8117
checkout paper-cpo-code https://github.com/Naoki04/paper-cpo-code.git d1597a4fc870124cef98922f737998bfabf8e6d4
checkout CAP-Dreamer https://github.com/ElliottDyson/CAP-Dreamer.git b39108d75a3bc01a972776f3f45a03914949470e
