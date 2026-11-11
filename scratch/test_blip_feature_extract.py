import time
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

print("Initializing test_blip_feature_extract...", flush=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

model_id = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)
model.eval()
print("Model loaded.", flush=True)

# Create dummy image and prompt
img = Image.new("RGB", (224, 224), color=(255, 255, 255))
prompt = "Question: What is 2 + 2? Correct answer: 4. Topic: Arithmetic."

inputs = processor(
    images=[img, img],
    text=[prompt, prompt],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).to(device)

with torch.no_grad():
    vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
    image_embeds = vision_outputs[0]
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    text_outputs = model.text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True
    )
    cls_repr = text_outputs.last_hidden_state[:, 0, :]
    print(f"CLS representation shape: {cls_repr.shape}", flush=True)
    print("Test passed successfully!", flush=True)

