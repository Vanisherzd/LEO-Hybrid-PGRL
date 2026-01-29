#!/bin/bash
# NASA Earthdata Downloader (Bash Fallback)

# 1. Credentials
NASA_USER="YOUR_USER"
NASA_PASS="YOUR_PASS"

# Load from .env if present
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

# 2. Setup Netrc for Curl
echo "machine urs.earthdata.nasa.gov login $NASA_USER password $NASA_PASS" > .netrc
echo "Creating .netrc file..."

# 3. Download
# Example URL ($1)
URL=$1
OUTPUT=$2

if [ -z "$URL" ]; then
  echo "Usage: ./get_sp3.sh <URL> [OUTPUT_FILENAME]"
  exit 1
fi

if [ -z "$OUTPUT" ]; then
  OUTPUT=$(basename "$URL")
fi

echo "Downloading $URL to $OUTPUT..."

# -L: Follow redirects
# -n: Use .netrc
# -b/-c: Cookie handling
curl -L -n -c cookies.txt -b cookies.txt -o "$OUTPUT" "$URL"

# 4. Cleanup
rm .netrc cookies.txt
echo "Done."
