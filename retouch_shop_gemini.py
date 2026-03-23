"""Gemini shop interior enhancement with safety filters disabled"""
import sys, os, time, traceback
from google import genai
from google.genai import types
from io import BytesIO
from PIL import Image

PROMPT = (
    "Enhance this upscale bar interior photo to feel more luxurious, premium, and atmospheric. "
    "Brighten the lighting slightly while preserving a natural night ambience, add a warmer color temperature, "
    "improve contrast and clarity, make highlights glow softly, deepen rich wood / gold / amber tones if present, "
    "and create a refined high-end lounge mood. Keep the composition, furniture layout, objects, signage, and perspective unchanged. "
    "Do not add or remove people or objects. Do not stylize it unrealistically. Aim for elegant, realistic architectural/interior photo enhancement."
)

SAFETY = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
]


def retouch(src_path, out_path):
    client = genai.Client()

    img = Image.open(src_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    img_bytes = buf.getvalue()

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            PROMPT,
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=SAFETY,
        )
    )

    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            out_img = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out_img.save(out_path, format="JPEG", quality=94)
            print(f"done: {out_path} ({os.path.getsize(out_path)} bytes)")
            return True
        elif getattr(part, "text", None):
            print(f"Model said: {part.text}")

    print(f"Error: No image generated for {src_path}")
    return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python retouch_shop_gemini.py input.jpg output.jpg [sleep_sec]")
        sys.exit(1)

    src_path = sys.argv[1]
    out_path = sys.argv[2]
    sleep_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0

    try:
        ok = retouch(src_path, out_path)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        sys.exit(0 if ok else 2)
    except Exception:
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
