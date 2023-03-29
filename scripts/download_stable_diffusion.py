from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import os


def download(model_dir: str = "models"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_id = "stabilityai/stable-diffusion-2"
    pipeline = StableDiffusionPipeline(model_id)
    scheduler = EulerDiscreteScheduler()
    pipeline.download(scheduler, os.path.join(model_dir, model_id))


if __name__ == "__main__":
    download()