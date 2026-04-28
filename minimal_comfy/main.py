from __future__ import annotations
import torch
import os
import sys
import math
import logging
import random
from PIL import Image

# Add the current directory to sys.path so comfy and other modules can be found
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

import folder_paths
import latent_preview
import node_helpers

MAX_RESOLUTION=16384

def set_base_directory(path: str):
    """Sets the base directory for all ComfyUI folders (models, input, output, etc.)."""
    path = os.path.abspath(path)
    folder_paths.base_path = path
    folder_paths.models_dir = os.path.join(path, "models")
    folder_paths.output_directory = os.path.join(path, "output")
    folder_paths.input_directory = os.path.join(path, "input")
    folder_paths.temp_directory = os.path.join(path, "temp")
    
    # Update existing folder mappings
    for name in folder_paths.folder_names_and_paths:
        paths, extensions = folder_paths.folder_names_and_paths[name]
        # Only update paths that were relative to the old base_dir (models folder)
        new_paths = [os.path.join(folder_paths.models_dir, os.path.basename(p)) for p in paths]
        folder_paths.folder_names_and_paths[name] = (new_paths, extensions)

def get_path(folder_name: str, filename: str) -> str:
    """Helper to get path, supporting absolute paths."""
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    return folder_paths.get_full_path_or_raise(folder_name, filename)

class CLIPTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."
    SEARCH_ALIASES = ["text", "prompt", "text prompt", "positive prompt", "negative prompt", "encode text", "text encoder", "encode prompt"]

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens), )

class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"
    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."
    SEARCH_ALIASES = ["decode", "decode latent", "latent to image", "render latent"]

    def decode(self, vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )

class VAELoader:
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]
    
    @staticmethod
    def vae_list(s):
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = sdxl_taesd_dec = sd1_taesd_enc = sd1_taesd_dec = sd3_taesd_enc = sd3_taesd_dec = f1_taesd_enc = f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."): sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."): sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."): sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."): sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."): sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."): sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."): f1_taesd_dec = True
            elif v.startswith("taef1_decoder."): f1_taesd_enc = True
            else:
                for tae in s.video_taes:
                    if v.startswith(tae): vaes.append(v)

        if sd1_taesd_dec and sd1_taesd_enc: vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc: vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc: vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc: vaes.append("taef1")
        vaes.append("pixel_space")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))
        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc: sd["taesd_encoder.{}".format(k)] = enc[k]
        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec: sd["taesd_decoder.{}".format(k)] = dec[k]
        if name == "taesd": sd["vae_scale"], sd["vae_shift"] = torch.tensor(0.18215), torch.tensor(0.0)
        elif name == "taesdxl": sd["vae_scale"], sd["vae_shift"] = torch.tensor(0.13025), torch.tensor(0.0)
        elif name == "taesd3": sd["vae_scale"], sd["vae_shift"] = torch.tensor(1.5305), torch.tensor(0.0609)
        elif name == "taef1": sd["vae_scale"], sd["vae_shift"] = torch.tensor(0.3611), torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(s), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    def load_vae(self, vae_name):
        metadata = None
        if vae_name == "pixel_space":
            sd = {"pixel_space_vae": torch.tensor(1.0)}
        elif vae_name in self.image_taes:
            sd = self.load_taesd(vae_name)
        else:
            if os.path.isabs(vae_name) and os.path.isfile(vae_name):
                vae_path = vae_name
            elif os.path.splitext(vae_name)[0] in self.video_taes:
                vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)

class UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"advanced": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn": model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2": model_options["dtype"] = torch.float8_e5m2

        unet_path = get_path("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image"], ),
                              },
                "optional": { "device": (["default", "cpu"], {"advanced": True}), }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        model_options = {}
        if device == "cpu": model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip_path = get_path("text_encoders", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)

class EmptyLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"
    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return ({"samples": latent, "downscale_ratio_spacial": 8}, )

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out, )

class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

NODE_CLASS_MAPPINGS = {
    "UNETLoader": UNETLoader,
    "CLIPLoader": CLIPLoader,
    "VAELoader": VAELoader,
    "CLIPTextEncode": CLIPTextEncode,
    "KSampler": KSampler,
    "VAEDecode": VAEDecode,
    "EmptyLatentImage": EmptyLatentImage,
}

# Instantiate the classes as requested
UNETLoader_instance = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader_instance = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader_instance = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode_instance = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler_instance = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode_instance = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage_instance = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
