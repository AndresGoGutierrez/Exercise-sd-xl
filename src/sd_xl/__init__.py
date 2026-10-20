import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype = torch.float32
    # FloatXX 32 0 16
    # Me indica la cantidad de decimales de precision
    # con el que el modelo creara la imagen
    # Si mi pc me genera una imagen negra, cambio a 16
)

modelo = modelo.to("cpu")
# cuda - nvidia
# mps - mac com procesadores mx
# xpu - Grafica Intel Iris

prompt = input("Escribe el prompt de la imagen: ")
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

print("Generando la imagen...")

imagen = modelo(
    prompt = prompt,
    negative_prompt = negative_prompt,
    num_inference_steps = 30, # Numero de veces que creara y corregira la imagen
    guidance_scale = 7,
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")
print("Imagen guardada")

