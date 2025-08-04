import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from .flux_kontext_example.flux_kontext_dit_teacache import forward as teacache_forward


@torch.no_grad()
def run():
    num_inference_steps = 28

    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipe.transformer.forward = teacache_forward

    # TeaCache
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.19  # thresh
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None

    pipe.to("cuda")
    input_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    )

    image = pipe(
        image=input_image,
        prompt="Add a hat to the cat",
        guidance_scale=2.5,
        num_inference_steps=num_inference_steps,
    ).images[0]


if __name__ == "__main__":
    run()
