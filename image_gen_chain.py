from datetime import datetime
from IPython.display import display, HTML
from PIL import Image
import time
import base64


class ImageGenChain:
    def __init__(
        self,
        inputs: dict[str, dict],
        steps: dict[str, dict],
        generate,
        image_grid,
        save_image_to_ram,
        preview_width: int = 100,
        direction: str = "TD",
        delay: int = 3,
    ):
        self.images = { "inputs": inputs }
        
        self.steps = steps
        self.generate = generate
        self.image_grid = image_grid
        self.save_image_to_ram = save_image_to_ram

        self.preview_width = preview_width
        self.direction = direction
        self.delay = delay

        self.running_idx = 0

        self.gen_id = f"igc_{datetime.now().strftime('%d%m%y_%H%M')}"

        print(f"Gen ID: {self.gen_id}")

        self.display = display(
            HTML(self._get_mermaid_html()),
            display_id=True
        )

        time.sleep(self.delay)

    # =========================
    # MERMAID
    # =========================
    def _get_mermaid_code(self):
        step_ids = [step["step_id"] for step in self.steps]

        edges = [
            (ref_img, step["step_id"])
            for step in self.steps
            for ref_img in step["ref_images"]
        ]

        nodes = ["inputs"]
        nodes += step_ids

        code = f"""
            flowchart {self.direction}
                classDef done fill:#dcfce7,color:#166534,stroke:#16a34a
                classDef running fill:#fef3c7,color:#92400e,stroke:#d97706

                {"\n    ".join(nodes)}

                {"\n    ".join([
                    f"{edge[0].split('.')[0]} --> {edge[1]}"
                    for edge in edges
                ])}
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
        ref_strings: list[str]
    ) -> list[Image] | None:

        ref_images: list[Image] = []

        for ref_string in ref_strings:

            if "." in ref_string:
                [step_id, image_id] = ref_string.split(".")

            else:
                step_id = ref_string

                step_image_keys = list(
                    self.images[step_id].keys()
                )

                step_images = [
                    self._get_image(step_id, k)
                    for k in step_image_keys
                ]

                if len(step_image_keys) == 1:
                    image_id = step_image_keys[0]

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
                        f"Choose image of step {step_id}"
                    )

                    print(
                        f"Choosen {step_id}.{image_id}"
                    )

            ref_images.append(
                self._get_image(step_id, image_id)
            )

        return ref_images or None

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

            self.display.update(
                HTML(self._get_mermaid_html())
            )

            time.sleep(self.delay)

            print(step)

            step_id = step["step_id"]

            ref_images = self._get_ref_images(
                step["ref_images"]
            )

            print("Ref images:")

            display(
                self.image_grid(
                    ref_images,
                    1,
                    len(ref_images),
                    preview_width=self.preview_width
                )
            )

            kwargs = step["configs"]

            kwargs["prompt"] = step["prompt"]
            kwargs["images"] = ref_images

            imgs = self.generate(**kwargs)

            self.images[step_id] = {}

            for idx, img in enumerate(imgs):

                self.images[step_id][f"{idx+1}"] = img

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

        self.display.update(
            HTML(self._get_mermaid_html())
        )

        time.sleep(self.delay)

        return self.images
    
