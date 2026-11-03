import os
import io
import gc
import math
import base64
import html
import pyzipper
import ipywidgets as widgets

from io import BytesIO
from PIL import Image
from IPython.display import display, HTML


class RAMImageManager:
    def __init__(self):
        self.IMAGES = {}
        self.upload = None
        self.new_upload()

    def new_upload(self):
        self.upload = widgets.FileUpload(accept="image/*", multiple=True)
        self.upload.observe(self.load_images_to_ram, names="value")
        display(self.upload)

    def load_images_to_ram(self, change):
        uploaded_files = self.upload.value
        
        # dict (ipywidgets v8)
        if isinstance(uploaded_files, dict):
            for filename, file_info in uploaded_files.items():
                content = file_info["content"]
                # Đọc byte trực tiếp vào RAM bằng BytesIO
                img = Image.open(io.BytesIO(content)).convert("RGB")
                self.save_image_to_ram(filename, img)
        
        # list/tuple (ipywidgets v7)
        elif isinstance(uploaded_files, (list, tuple)):
            for file_info in uploaded_files:
                filename = file_info["name"]
                content = file_info["content"]
                img = Image.open(io.BytesIO(content)).convert("RGB")
                self.save_image_to_ram(filename, img)
        
        print(f"RAM IMAGES: {len(self.IMAGES)}")

    def save_image_to_ram(self, filename, image):
        if filename in self.IMAGES:
            new_filename = filename
            counter = 1

            while new_filename in self.IMAGES:
                new_filename = (
                    f"{filename.rsplit('.', 1)[0]}_"
                    f"{counter}."
                    f"{filename.rsplit('.', 1)[1]}"
                )
                counter += 1

            filename = new_filename
        self.IMAGES[filename] = image
        print(
            f"ADD RAM IMAGE: "
            f"IMAGES['{filename}'] "
            f"({image.size[0]}x{image.size[1]})"
        )
        return filename

    def list_all_ram_images(self):
        return list(self.IMAGES.keys())

    def display_all_ram_images(self, max_col=6, preview_width=50):
        if not self.IMAGES:
            print("No images in RAM.")
            return

        cols = min(max_col, len(self.IMAGES))
        items = []

        for filename, img in self.IMAGES.items():
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()

            items.append(f"""
                <div style="text-align:center; min-width:0;">
                    <img src="data:image/png;base64,{encoded}"
                         style="width:100%; height:auto;">
                    <div style="margin-top:6px; overflow-wrap:anywhere;
                                font-size:12px;">
                        {html.escape(str(filename))}
                    </div>
                </div>
            """)

        display(HTML(f"""
            <div style="display:grid;
                        grid-template-columns:repeat({cols}, {preview_width}px);
                        gap:16px;">
                {''.join(items)}
            </div>
        """))

    def download_ram_images_as_zip(
        self,
        keys=None,
        password="123456",
        zip_filename="backup_images.zip"
    ):
        images_dict = self.IMAGES

        if not images_dict:
            print("NO IMAGES in RAM to save!")
            return

        target_keys = (keys if keys is not None else list(images_dict.keys()))
        zip_buffer = io.BytesIO()

        # pyzipper.WZ_AES
        with pyzipper.AESZipFile(zip_buffer, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
            zf.setpassword(password.encode("utf-8"))
            saved_count = 0
            for key in target_keys:
                if key in images_dict:
                    img_obj = images_dict[key]
                    # Byte Stream
                    img_bytes_io = io.BytesIO()
                    ext = (key.split(".")[-1].upper() if "." in key else "PNG")
                    fmt = ("JPEG" if ext in ["JPG", "JPEG"] else "PNG")
                    img_obj.save(img_bytes_io, format=fmt)
                    # ZIP
                    zf.writestr(key, img_bytes_io.getvalue())
                    saved_count += 1
                else:
                    print(
                        f"Warning: Key '{key}' "
                        f"not found in images_dict. Skipping."
                    )
        if saved_count == 0:
            print("There are no valid images to compress!")
            return

        zip_bytes = zip_buffer.getvalue()
        b64_zip = base64.b64encode(zip_bytes).decode("utf-8")

        html_code = f"""
        <div style="margin:15px 0;">
            <p style="margin-bottom:5px; color:#28a745;
                      font-weight:bold;">
                Saved {saved_count} images in RAM
                (Password: <code>{html.escape(password)}</code>)
            </p>

            <a download="{html.escape(zip_filename, quote=True)}"
               href="data:application/zip;base64,{b64_zip}"
               target="_blank">

                <button style="padding:10px 20px;
                               background-color:#007bff;
                               color:white;
                               border:none;
                               border-radius:5px;
                               cursor:pointer;
                               font-weight:bold;">
                    DOWNLOAD ({html.escape(zip_filename)})
                </button>

            </a>
        </div>
        """

        display(HTML(html_code))
        
