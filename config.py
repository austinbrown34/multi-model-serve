from scripts.download_gptj import download as gptj_download
from scripts.download_led_summary import download as led_summary_download
from scripts.download_stable_diffusion import download as stable_diffusion_download


SELECTED_MODELS = {
    "gptj": {
        "download": gptj_download
    },
    "led_summary": {
        "download": led_summary_download
    },
    "stable_diffusion": {
        "download": stable_diffusion_download
    }
}