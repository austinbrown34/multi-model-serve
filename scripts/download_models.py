from download_gptj import download as gptj_download
from download_led_summary import download as led_summary_download
from download_stable_diffusion import download as stable_diffusion_download


def download_models():
    gptj_download()
    led_summary_download()
    stable_diffusion_download()
    

if __name__ == "__main__":
    download_models()