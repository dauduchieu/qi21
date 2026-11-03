from pydantic import BaseModel
from openai import OpenAI

from PIL import Image
from io import BytesIO
import base64


class OpenAILLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.model = model

    # =========================
    # IMAGE ENCODING
    # =========================

    def _image_to_data_url(
        self,
        image: Image.Image,
        image_format: str = "JPEG",
        quality: int = 90,
    ) -> str:
        """
        Convert PIL Image in RAM to Base64 Data URL.
        """

        buffer = BytesIO()

        if image_format.upper() == "JPEG":
            # JPEG không hỗ trợ alpha channel
            image = image.convert("RGB")

            image.save(
                buffer,
                format="JPEG",
                quality=quality,
            )

            mime_type = "image/jpeg"

        elif image_format.upper() == "PNG":
            image.save(
                buffer,
                format="PNG",
            )

            mime_type = "image/png"

        else:
            raise ValueError(
                f"Unsupported image format: {image_format}"
            )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:{mime_type};base64,{image_base64}"

    def _build_user_content(
        self,
        user_prompt: str,
        images: list[Image.Image] | Image.Image | None = None,
    ) -> list[dict]:
        """
        Build multimodal user content.

        Supports:
            - Text only
            - One PIL Image
            - List of PIL Images
        """

        content = [
            {
                "type": "text",
                "text": user_prompt,
            }
        ]

        if images is None:
            return content

        # Cho phép truyền một ảnh đơn
        if isinstance(images, Image.Image):
            images = [images]

        for image in images:

            if not isinstance(image, Image.Image):
                raise TypeError(
                    "Each item in images must be a PIL Image."
                )

            image_url = self._image_to_data_url(image)

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            })

        return content

    # =========================
    # LLM CALL
    # =========================

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[Image.Image] | Image.Image | None = None,
        extra_body: dict | None = None,
        max_completion_tokens: int = 16384,
        OutputModel: type[BaseModel] | None = None,
    ) -> str | BaseModel:

        if extra_body is None:
            extra_body = {}

        user_content = self._build_user_content(
            user_prompt=user_prompt,
            images=images,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # =========================
        # STRUCTURED OUTPUT
        # =========================

        if OutputModel is not None:

            response = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=OutputModel,
                max_completion_tokens=max_completion_tokens,
                extra_body=extra_body,
            )

            message = response.choices[0].message

            if message.refusal:
                raise ValueError(
                    f"Model refused to answer: {message.refusal}"
                )

            if message.parsed is None:
                raise ValueError(
                    "Model did not return a valid structured output."
                )

            return message.parsed

        # =========================
        # NORMAL TEXT OUTPUT
        # =========================

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
        )

        choice = response.choices[0]
        message = choice.message
        content = message.content

        if choice.finish_reason != "stop":
            raise ValueError(
                "Model response is incomplete. "
                f"Finish reason: {choice.finish_reason}"
            )

        if content is None:
            raise ValueError(
                "Model returned an empty response."
            )

        return content
    
    