# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline
from settings import settings
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=settings.HUGGINGFACE_TOKEN)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt)["sample"][0]  # needs 51 (took 30 min)

    image.save("astronaut_rides_horse.png")

    num_images = 3
    prompt = ["a photograph of an lobster raviolo with dill, sage and lemon served by an italian chef"] * num_images

    images = pipe(prompt)["sample"]

    grid = image_grid(images, rows=1, cols=3)


if __name__ == "__main__":
    main()


"""
%load_ext autoreload
%autoreload 2
"""
