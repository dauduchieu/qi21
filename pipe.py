import os, gc, math, torch
from PIL import Image

from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
    AutoencoderKLQwenImage21,
)


def load_qwen_pipeline(
    gen_gpu: int = 0, gen_gpu_gb: int = 6, gen_cpu_gb: int = 4,
    txt_gpu: int = 1, txt_gpu_gb: int = 6, txt_cpu_gb: int = 4,
    turbo:bool=False,
):
    ######### CONFIG #########
    MODEL_ID = "Qwen/Qwen-Image-2.1"
    default_step = 40
    
    if turbo:
        MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
        default_step = 8

    DTYPE = torch.float16

    GEN_GPU = gen_gpu
    TEXT_GPU = txt_gpu

    MAX_GEN = { GEN_GPU: f"{gen_gpu_gb}GiB", "cpu": f"{gen_cpu_gb}GiB" }
    MAX_TXT = { TEXT_GPU: f"{txt_gpu_gb}GiB", "cpu": f"{txt_cpu_gb}GiB" }

    OFFLOAD_GEN = "/tmp/qwen_gen"
    OFFLOAD_TXT = "/tmp/qwen_txt"

    os.makedirs(OFFLOAD_GEN, exist_ok=True)
    os.makedirs(OFFLOAD_TXT, exist_ok=True)

    ######### DOWNLOAD #########
    local = snapshot_download(MODEL_ID)

    tr_dir = os.path.join(local, "transformer")
    vae_dir = os.path.join(local, "vae")
    te_dir = os.path.join(local, "text_encoder")
    proc_dir = os.path.join(local, "processor")

    if not os.path.isdir(proc_dir):
        proc_dir = local

    print(local)

    ######### TRANSFORMER #########
    config = QwenImage21Transformer2DModel.load_config(tr_dir)

    with init_empty_weights():
        transformer = QwenImage21Transformer2DModel.from_config(config)

    no_split = list({ type(m).__name__ for m in transformer.modules() if "Block" in type(m).__name__ })

    print(no_split)

    device_map = infer_auto_device_map(
        transformer,
        max_memory=MAX_GEN,
        no_split_module_classes=no_split,
        dtype=DTYPE,
    )

    print(device_map)

    transformer = load_checkpoint_and_dispatch(
        transformer,
        checkpoint=tr_dir,
        device_map=device_map,
        offload_folder=OFFLOAD_GEN,
        offload_state_dict=True,
        dtype=DTYPE,
        no_split_module_classes=no_split,
    )

    transformer.eval()

    ######### PROCESSOR #########
    processor = Qwen3VLProcessor.from_pretrained(proc_dir)

    ######### TEXT ENCODER #########
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        te_dir,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=MAX_TXT,
        offload_folder=OFFLOAD_TXT,
        offload_state_dict=True,
        attn_implementation="sdpa",
    )

    text_encoder.eval()

    # print(text_encoder.hf_device_map)

    ######### VAE #########
    vae = AutoencoderKLQwenImage21.from_pretrained(
        vae_dir,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(f"cuda:{GEN_GPU}")

    vae.enable_slicing()

    vae.enable_tiling(
        tile_sample_min_height=512,
        tile_sample_min_width=512,
        tile_sample_stride_height=256,
        tile_sample_stride_width=256,
    )

    vae.eval()

    ######### PIPE #########
    pipe = QwenImage21Pipeline.from_pretrained(
        local,
        text_encoder=None,
        processor=processor,
        transformer=None,
        vae=None,
        torch_dtype=DTYPE,
    )

    return pipe, transformer, processor, text_encoder, vae, default_step


def encode_mode(pipe, text_encoder):
    pipe.text_encoder = text_encoder
    pipe.transformer = None
    pipe.vae = None


def generate_mode(pipe, transformer, vae):
    pipe.text_encoder = None
    pipe.transformer = transformer
    pipe.vae = vae


def calc_size(area, ratio):
    w = math.sqrt(area * ratio)
    h = w / ratio

    return round(w / 32) * 32, round(h / 32) * 32


def prepare_images(pipe, images, resolution=1024):
    if images is None: return None
    if not isinstance(images, list): images = [images]
    out = []
    for img in images:
        if not isinstance(img, Image.Image): img = Image.fromarray(img)
        img = img.convert("RGBA")
        w, h = img.size
        rw, rh = calc_size(resolution**2, w / h)
        out.append( pipe.image_processor.resize(img, width=rw, height=rh) )
    return out


@torch.inference_mode()
def encode(
    pipe,
    text_encoder,
    txt_gpu,
    gen_gpu,
    prompt,
    images=None,
):
    encode_mode(pipe, text_encoder)

    x = pipe.encode_prompt(
        prompt=prompt,
        image=images,
        device=f"cuda:{txt_gpu}",
        num_images_per_prompt=1,
    )

    return tuple(t.to(f"cuda:{gen_gpu}") if t is not None else None for t in x)


@torch.inference_mode()
def generate(
    pipe, transformer, text_encoder, vae,
    txt_gpu, gen_gpu,
    prompt,
    images=None,
    width=None, height=None,
    steps=40,
    seed=42,
    cfg=1.0,
    negative_prompt=None,
    resolution=1024,
    n_images=1,
):
    images = prepare_images(pipe, images, resolution)
    if images and (width is None or height is None):
        w, h = images[0].size
        auto_w, auto_h = calc_size(resolution**2, w / h)
        width = width or auto_w
        height = height or auto_h

    print("> Encode")
    pos, pos_mask, pos_img_mask = encode(
        pipe,
        text_encoder,
        txt_gpu,
        gen_gpu,
        prompt,
        images,
    )

    neg = neg_mask = neg_img_mask = None

    if cfg > 1 and negative_prompt:
        neg, neg_mask, neg_img_mask = encode(
            pipe,
            text_encoder,
            txt_gpu,
            gen_gpu,
            negative_prompt,
            images,
        )

    print("> Generate")

    generate_mode(pipe, transformer, vae)

    saved_masks = { pos.data_ptr(): pos_img_mask }

    if neg is not None:
        saved_masks[neg.data_ptr()] = neg_img_mask

    original_encode = pipe.encode_prompt

    def patched_encode(*args, **kwargs):
        emb = kwargs.get("prompt_embeds")

        if emb is not None and kwargs.get("image_pad_mask") is None:
            mask = saved_masks.get(emb.data_ptr())

            if mask is None:
                mask = emb.new_zeros(
                    emb.shape[:2],
                    dtype=torch.bool,
                )

            kwargs["image_pad_mask"] = mask

        return original_encode(*args, **kwargs)

    pipe.encode_prompt = patched_encode

    try:
        result = pipe(
            prompt_embeds=pos,
            prompt_embeds_mask=pos_mask,
            negative_prompt_embeds=neg,
            negative_prompt_embeds_mask=neg_mask,
            image=images,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            output_resolution=resolution,
            num_images_per_prompt=n_images,
            use_kv_cache=False,
            generator=torch.Generator(
                device=f"cuda:{gen_gpu}"
            ).manual_seed(seed),
        )

    finally:
        pipe.encode_prompt = original_encode

    return result.images

