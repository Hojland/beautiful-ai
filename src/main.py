# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline
from settings import settings


def main():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=settings.HUGGINGFACE_TOKEN)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt)["sample"][0]

    image.save("astronaut_rides_horse.png")


if __name__ == "__main__":
    main()


"""
%load_ext autoreload
%autoreload 2
"""
