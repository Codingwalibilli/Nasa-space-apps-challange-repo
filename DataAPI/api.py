from flask import Flask, request, jsonify
import os
import math
import requests
import numpy as np
import re
import cv2

app = Flask(__name__)

OUTPUT_DIR = "data/processed_dzi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_SIZE = 256
OVERLAP = 0
FORMAT = "jpg"


def download_img_bytes(url):
    print(f"📥 Downloading IMG from: {url}")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Failed to download {url} (status {r.status_code})")
    print(f"✅ Download complete, size: {len(r.content)} bytes")
    return r.content


def extract_pds3_metadata(img_bytes, max_bytes=8192):
    print("🔍 Extracting PDS3 metadata from header...")
    header_bytes = img_bytes[:max_bytes]
    header_text = header_bytes.decode("ascii", errors="ignore")
    metadata = {}
    for match in re.finditer(r'(\^?\w+)\s*=\s*"?([^"\n]+)"?', header_text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        metadata[key] = value
    for k in ['LINES', 'LINE_SAMPLES', 'SAMPLE_BITS']:
        if k in metadata:
            metadata[k] = int(metadata[k])
    print(f"📄 Metadata extracted: {metadata}")
    return metadata


def read_ctx_image(img_bytes, metadata):
    lines = metadata['LINES']
    samples = metadata['LINE_SAMPLES']
    print(f"🖼 Reading raw image: {lines}x{samples}")

    def find_image_start(img_bytes):
        ascii_part = img_bytes[:32768].decode("ascii", errors="ignore")
        end = ascii_part.find("END")
        if end != -1:
            return end + 3  
        return 8192  

    header_size = find_image_start(img_bytes)
    raw_bytes = img_bytes[header_size:]
    
    bytes_per_pixel = len(raw_bytes) / (lines * samples)

    if abs(bytes_per_pixel - 1) < 0.1:
        dtype = np.uint8
        endian = '<'  
    elif abs(bytes_per_pixel - 2) < 0.1:
        dtype = np.uint16
        endian = '>' 
    else:
        raise ValueError(f"Unexpected bytes per pixel: {bytes_per_pixel}")

    print(f"📏 Detected {bytes_per_pixel:.2f} bytes/pixel, dtype={dtype}, endian={endian}")

    # --- Read array properly ---
    img_array = np.frombuffer(raw_bytes, dtype=np.dtype(endian + np.dtype(dtype).char))
    img_array = img_array[:lines * samples]
    img_array = img_array.reshape((lines, samples))

    img_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)
    print(f"✅ Image array ready: shape {img_norm.shape}, dtype={img_norm.dtype}")
    return img_norm

def generate_dzi(image, output_base):
    h, w = image.shape[:2]
    max_level = math.ceil(math.log2(max(w, h)))
    print(f"⚡ Generating DZI: {w}x{h}, levels: {max_level + 1}")

    dzi_dir = output_base + "_files"
    os.makedirs(dzi_dir, exist_ok=True)

    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        lvl_w, lvl_h = math.ceil(w / scale), math.ceil(h / scale)
        lvl_img = cv2.resize(image, (lvl_w, lvl_h), interpolation=cv2.INTER_AREA)

        level_dir = os.path.join(dzi_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)

        for y in range(0, lvl_h, TILE_SIZE):
            for x in range(0, lvl_w, TILE_SIZE):
                tile = lvl_img[y:y + TILE_SIZE, x:x + TILE_SIZE]
                tile_path = os.path.join(level_dir, f"{x//TILE_SIZE}_{y//TILE_SIZE}.{FORMAT}")
                cv2.imwrite(tile_path, tile, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"   - Level {level} done, size: {lvl_w}x{lvl_h}")

    dzi_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image TileSize="{TILE_SIZE}" Overlap="{OVERLAP}" Format="{FORMAT}" xmlns="http://schemas.microsoft.com/deepzoom/2008">
    <Size Width="{w}" Height="{h}"/>
</Image>"""
    dzi_path = f"{output_base}.dzi"
    with open(dzi_path, "w") as f:
        f.write(dzi_xml)

    print(f"✅ DZI generation complete: {dzi_path}")
    return dzi_path


@app.route("/generate_dzi", methods=["POST"])
def generate_dzi_api():
    data = request.json
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    url = data["url"].strip()
    base_name = os.path.splitext(os.path.basename(url))[0]
    output_base = os.path.join(OUTPUT_DIR, base_name)
    dzi_path = f"{output_base}.dzi"

    try:
        if os.path.exists(dzi_path):
            print(f"♻️ Using cached DZI: {dzi_path}")
            return jsonify({"dzi_path": dzi_path, "metadata": {}})

        img_bytes = download_img_bytes(url)
        metadata = extract_pds3_metadata(img_bytes)
        img_array = read_ctx_image(img_bytes, metadata)
        dzi_path = generate_dzi(img_array, output_base)

        return jsonify({"dzi_path": dzi_path, "metadata": metadata})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Flask API running at http://localhost:5000")
    app.run(debug=True, port=5000)
