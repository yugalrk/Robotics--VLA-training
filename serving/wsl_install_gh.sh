#!/bin/bash
# Download the official GitHub CLI release (github.com/cli/cli), verify its SHA-256 against the release's
# checksums file, install the gh binary to ~/.local/bin. No sudo.
set -e
TAG=$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest | grep -m1 '"tag_name"' | cut -d'"' -f4)
VER=${TAG#v}
NAME=gh_${VER}_linux_amd64.tar.gz
DL=$(mktemp -d ~/gh_download.XXXXXX)
cd "$DL"
curl -fsSLO "https://github.com/cli/cli/releases/download/${TAG}/${NAME}"
curl -fsSLO "https://github.com/cli/cli/releases/download/${TAG}/gh_${VER}_checksums.txt"
echo "downloaded $NAME ($(du -h "$NAME" | cut -f1)) from github.com/cli/cli release $TAG"
grep " ${NAME}\$" "gh_${VER}_checksums.txt" | sha256sum -c -
tar -xzf "$NAME"
mkdir -p ~/.local/bin
cp "gh_${VER}_linux_amd64/bin/gh" ~/.local/bin/gh
chmod +x ~/.local/bin/gh
cd ~ && rm -rf "$DL"
~/.local/bin/gh --version | head -1
