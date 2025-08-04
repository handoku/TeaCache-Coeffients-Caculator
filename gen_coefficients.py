from functools import partial
import json


import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from .flux_kontext_example.transformer_flux_forward import (
    forward as transformer_flux_forward,
)
from .teacache_caculator import TeaCacheCoefficientCaculator


@torch.no_grad()
def get_coefficients():
    num_inference_steps = 28

    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    calculator = TeaCacheCoefficientCaculator(num_inference_steps)
    forward_func = partial(transformer_flux_forward, calculator=calculator)
    pipe.transformer.forward = forward_func
    pipe.to("cuda")

    with open("./data/dataset.json", "r") as f:
        dataset = json.load(f)
    
    for i, item in enumerate(dataset):
        img_url = item["url"]
        prompt = item["edit"]
        input_image = load_image(img_url)
        image = pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=2.5,
            num_inference_steps=num_inference_steps,
        ).images[0]
    coefficients = calculator.calculate_coefficients()
    return coefficients


if __name__ == "__main__":
    coefficients = get_coefficients()
    print(coefficients)
