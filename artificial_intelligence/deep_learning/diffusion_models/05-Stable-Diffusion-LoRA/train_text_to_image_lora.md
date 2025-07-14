# 如何基于Stable Diffusion LoRA进行finetune

```
huggingface-cli login

git clone https://github.com/huggingface/diffusers.git
pip install accelerate
pip install git+https://github.com/huggingface/diffusers

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
#export DATASET_NAME="atasoglu/flickr8k-dataset"
export DATASET_NAME="fantasyfish/laion-art"
#export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  ./examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
#export DATASET_NAME="atasoglu/flickr8k-dataset"
export DATASET_NAME="fantasyfish/laion-art"
#export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch ./examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337

https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main

wget -P data/ https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
wget -P data/ https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

huggingface-cli login

--train_data_dir="./data" \需要请求权限
开源库中有diffusers\examples\text_to_image\README.md 有说明怎么用diffusers库加载预训练模型和lora模型进行推理
```