# TeaCache-Coeffients-Caculator

This is a tool/tutorial for calculating the rescaling polynomial coefficients of TeaCache.

[TeaCache](https://github.com/ali-vilab/TeaCache) is an amazing training-free caching approach for accelerating diffusion transformer model inference, which achieves remarkable speedup with negligible image/video quality loss.

However, it is not that trivial to apply it to your own model for its magic coefficients parameters.

This tool is designed to ease the pain of determining which feature would be better for caching `hidden_states residuals`, and calculating the rescaling coefficients automatically.

## Usage

Use [Flux.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) model as an example. I presume you have already installed the latest diffusers and downloaded the Flux Kontext model.

First, we copy the flux transformer forward function from diffusers and modify it to add a hook(our calculator) to collect `features` and `hidden_states residuals`. (Check the `transformer_flux_forward.py`, you can see a few lines of code I added, you may want to capture other features, call `calculator.add_feature` at where you want to)

Then, we should pick up some typical user cases as a dataset( which I don't have) for feature sampling, so I use some data from [Intruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) for the example. If you have your own user cases( which you should), you can use it to collect dedicated `features` and `hidden_states residuals`.

We will run the Flux Kontext pipeline as follows：

```bash
python gen_coefficients.py
```

Check `gen_coefficients.py`, you can see how to use the calculator to generate the coefficients.

Finally, you can use the coefficients to replace the flux coefficients in `flux_kontext_dit_teacache.py` to accelerate your model inference.

```bash
python inference.py
```

Lastly, I may have misunderstood the TeaCache mechanism. If you have any questions, please open an issue or contribute a PR. Thanks.

## Acknowledgement

Much thanks to [TeaCache](https://github.com/ali-vilab/TeaCache) for the great work, [Flux.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) for the great model, [Intruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) for the great dataset.
