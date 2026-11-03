from datetime import datetime
from IPython.display import display, HTML
from PIL import Image
import time
import base64


class ImageGenChain:
    def __init__(
        self,
        inputs: dict[str, dict],
        steps: list[dict],
        generate,
        image_grid,
        save_image_to_ram,
        gen_ref_image_kwarg: str = "images",
        preview_width: int = 100,
        direction: str = "TD",
        delay: int = 3,
    ):
        # =========================
        # CONFIG
        # =========================

        self.images = {"inputs": inputs}

        self.steps = steps
        self.generate = generate
        self.image_grid = image_grid
        self.save_image_to_ram = save_image_to_ram

        # Tên kwargs mà generate() dùng để nhận reference images
        self.gen_ref_image_kwarg = gen_ref_image_kwarg

        self.preview_width = preview_width
        self.direction = direction
        self.delay = delay

        self.running_idx = 0

        self.gen_id = f"igc_{datetime.now().strftime('%d%m%y_%H%M')}"

        print(f"Gen ID: {self.gen_id}")

        # =========================
        # INITIAL MERMAID DISPLAY
        # =========================

        self.display = display(
            HTML(self._get_mermaid_html()),
            display_id=True
        )

        time.sleep(self.delay)

    # =========================
    # MERMAID
    # =========================

    def _get_mermaid_code(self):
        step_ids = [
            step["step_id"]
            for step in self.steps
        ]

        edges = [
            (ref_img, step["step_id"])
            for step in self.steps
            for ref_img in step.get("ref_images", [])
        ]

        nodes = ["inputs"]
        nodes += step_ids

        nodes_text = "\n    ".join(nodes)

        edges_text = "\n    ".join([
            f"{edge[0].split('.')[0]} --> {edge[1]}"
            for edge in edges
        ])

        code = f"""
            flowchart {self.direction}
                classDef done fill:#dcfce7,color:#166534,stroke:#16a34a
                classDef running fill:#fef3c7,color:#92400e,stroke:#d97706

                {nodes_text}

                {edges_text}
        """

        if self.running_idx < len(self.steps):
            code += f"""
                class inputs,{",".join(step_ids[:self.running_idx])} done
                class {step_ids[self.running_idx]} running
            """
        else:
            code += f"""
                class inputs,{",".join(step_ids)} done
            """

        return code

    def _get_mermaid_html(self):
        code = self._get_mermaid_code()

        encoded = base64.urlsafe_b64encode(
            code.encode("utf-8")
        ).decode("ascii")

        return f"""
            <div style="width:300px; max-width:100%; height:auto; background-color:#ffffff;">
                <img
                    src="https://mermaid.ink/svg/{encoded}"
                    style="width:100%; max-width:100%; height:auto;"
                />
            </div>
        """

    # =========================
    # IMAGE MANAGEMENT
    # =========================

    def _get_image(
        self,
        step_id: str,
        image_id: str
    ) -> Image:

        try:
            return self.images[step_id][image_id]

        except:
            default_id = list(
                self.images[step_id].keys()
            )[0]

            print(
                f"\033[91mCan not get image: "
                f"{step_id}.{image_id}\033[0m. "
                f"Default: {step_id}.{default_id}"
            )

            return self.images[step_id][default_id]

    def _get_ref_images(
        self,
        ref_strings: list[str] | None
    ) -> list[Image] | None:

        if not ref_strings:
            return None

        ref_images: list[Image] = []

        for ref_string in ref_strings:

            # Reference có dạng: "step_id.image_id"
            if "." in ref_string:
                step_id, image_id = ref_string.split(".", 1)

            # Reference chỉ có step_id
            else:
                step_id = ref_string

                step_image_keys = list(
                    self.images[step_id].keys()
                )

                step_images = [
                    self._get_image(step_id, key)
                    for key in step_image_keys
                ]

                # Step chỉ có một ảnh → tự chọn
                if len(step_image_keys) == 1:
                    image_id = step_image_keys[0]

                # Step có nhiều ảnh → cho người dùng chọn
                else:
                    print(f"Choose {step_id}:")

                    display(
                        self.image_grid(
                            step_images,
                            1,
                            len(step_images),
                            preview_width=self.preview_width
                        )
                    )

                    print(f"Image key: {step_image_keys}")

                    image_id = input(
                        f"Choose image of step {step_id}: "
                    )

                    print(
                        f"Chosen {step_id}.{image_id}"
                    )

            ref_images.append(
                self._get_image(step_id, image_id)
            )

        return ref_images or None

    # =========================
    # GENERATE KWARGS
    # =========================

    def _get_generate_kwargs(
        self,
        step: dict
    ) -> dict:

        # step_id và ref_images do ImageGenChain xử lý riêng.
        # Tất cả key còn lại được truyền vào generate().
        kwargs = {
            key: value
            for key, value in step.items()
            if key not in ("step_id", "ref_images")
        }

        # Resolve reference images từ self.images
        ref_images = self._get_ref_images(
            step.get("ref_images")
        )

        # Chỉ thêm kwargs ảnh nếu step có ref_images
        if ref_images is not None:
            kwargs[self.gen_ref_image_kwarg] = ref_images

        return kwargs

    # =========================
    # SAVE IMAGES
    # =========================

    def save_images(self):

        image_keys = []

        for step_id in self.images.keys():

            for image_id in self.images[step_id].keys():

                image_key = (
                    f"{self.gen_id}.{step_id}.{image_id}.png"
                )

                self.save_image_to_ram(
                    image_key,
                    self._get_image(step_id, image_id)
                )

                image_keys.append(image_key)

        return image_keys

    # =========================
    # RUN PIPELINE
    # =========================

    def run(self):

        for step in self.steps:

            # Update Mermaid trước khi chạy step
            self.display.update(
                HTML(self._get_mermaid_html())
            )

            time.sleep(self.delay)

            print(step)

            step_id = step["step_id"]

            # Resolve reference images
            ref_images = self._get_ref_images(
                step.get("ref_images")
            )

            # Preview reference images nếu có
            if ref_images:

                print("Ref images:")

                display(
                    self.image_grid(
                        ref_images,
                        1,
                        len(ref_images),
                        preview_width=self.preview_width
                    )
                )

            # Lấy toàn bộ kwargs truyền vào generate()
            kwargs = self._get_generate_kwargs(step)

            # Generate images
            imgs = self.generate(**kwargs)

            # Lưu ảnh theo step_id
            self.images[step_id] = {}

            for idx, img in enumerate(imgs):

                self.images[step_id][f"{idx + 1}"] = img

            # Preview kết quả
            print("Result images:")

            display(
                self.image_grid(
                    imgs,
                    1,
                    len(imgs),
                    preview_width=self.preview_width
                )
            )

            self.running_idx += 1

        # Update Mermaid lần cuối khi pipeline hoàn tất
        self.display.update(
            HTML(self._get_mermaid_html())
        )

        time.sleep(self.delay)

        return self.images
    
