# TeaCache-Coeffients-Caculator

This is a tool/tutorial for calculating the polynomial coefficients of TeaCache.

[TeaCache] is a amazing training-free caching approach for accelerating diffusion transformer model inference. Which acheaves remarkable speedup with negligible image/video quality loss.

Howerever, it is not that trival to apply it to your own new model for its magic coefficients parameters.

This tool is designed to ease the pain of finding which feature is more important for caching hidden_states residuals, and to calculate the rescaling coefficients automatically.

## Usage

Use flux.1-kontext model as an example.

I presume you have already installed latest diffusers and downloaded the flux kontext model.

First, we copy the flux transformer forward function from diffusers, and modify it to add a hook(our calculator) to collect features and hidden_states residuals.(check the `transformer_flux_forward.py`, and you can see lines I added, you may want to capture other features)

Then, we should pick up some typical user cases as dataset( which I don't have) for feature sampling,, so I use intruct-pix2pix as an example. If you have your own user cases( which you should), you can use it to collect dedicated features and hidden_states residuals.

We will run flux kontext pipeline like follows：

```bash
python gen_coefficients.py
```

check `gen_coefficients.py`, you can see how to use the calculator to generate the coefficients.

finally, you can use the coefficients replace the flux coefficients in `flux_kontext_dit_teacache.py` to accelerate your model inference.

```bash
python inference.py
```

lastly, I may misunderstood the TeaCache mechanism, if you have any question, please open an issue or contribute a PR, thanks.

## Acknowledgement

Much thanks to [TeaCache](https://github.com/ali-vilab/TeaCache) for the great work, [Flux.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) for the great model, [Intruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) for the great dataset.
