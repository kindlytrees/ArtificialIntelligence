# pip install torch transformers safetensors diffusers
# huggingface-cli login

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# 加载基础模型
#base_model_id = "CompVis/stable-diffusion-v1-5"
base_model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")

# 加载 LoRA 权重
lora_weights = load_file("NSXNA1_v1.safetensors")
# 将 LoRA 权重应用到基础模型中
def apply_lora_weights(pipe, lora_weights):
    for name, param in pipe.unet.named_parameters():
        if name in lora_weights:
            param.data += lora_weights[name].data

apply_lora_weights(pipe, lora_weights)

prompts = [
    "bamboo drawed by zhengbanqiao with Lithography on several blank space",
    "pine drawed by zhengbanqiao",
    "the rolling hills draw by zhengbanqiao",
    "a branch of flower, traditional chinese ink painting by badashanren"
]

# 生成并保存多张图片
for i, prompt in enumerate(prompts):
    images = pipe(prompt, num_images_per_prompt=1).images
    for j, image in enumerate(images):
        image.save(f"output_image_{i+1}.png")

print("Images generated and saved.")


# # 推理示例
# prompt = "a nsxna1 vehicle near the river, the diver stand by the car window"
# image = pipe(prompt).images[0]

# # 保存生成的图像
# image.save("output_image.png")

# print("Image generated and saved as output_image.png")
# 定义多个 prompts
# prompts = [
#     "a nsxna1 vehicle near the river, the diver stand by the car window",
#     "a nsxna1 vehicle near the mountain, the diver stand by the car window",
#     "a nsxna1 car near the tree, the diver stand by the vehicle window",
#     "a nsxna1 car near the house, the diver is smoking on the seat"
# ]

# prompts = [
#     "bamboo drawed by zhengbanqiao with a chinese poem",
#     "a sunset scene near the sea in the style of wuchangshuo",
#     "bamboo drawed by zhengbanqiao with a chinese poem",
#     "a sunset scene near the sea in the style of wuchangshuo"
# ]
# a branch of flower, traditional chinese ink painting