import torch
from PIL import Image
import torchvision.transforms as T
import json
import os
from model_freezed import VGG16_piano

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

device = "cuda:0" if torch.cuda.is_available() else "cpu"


model = VGG16_piano().to(device)
model.load_state_dict(torch.load("models/teapot_vase_f1score_maro_vgg16.pth"))

def create_image_embeddings(model, data_root, class_name, metadata_file="embeddings_metadata.json"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    metadata = []

    with torch.no_grad():
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith((".jpg", ".png")):
                    image_path = os.path.join(root, file)
                    try:

                    
                        image = Image.open(image_path).convert("RGB")
                        image = transform(image).unsqueeze(0).to(device)

                        output= model(image)
                        predicted_class = torch.argmax(output, dim=1).item()

                        if predicted_class == class_name:
                            embedding = model.vgg.features(image).cpu().numpy().flatten().tolist()

                            metadata.append({
                             "image_path": image_path,
                             "embedding": embedding
                            })
                    except (OSError, IOError) as e:
                        print(f"❌ Fehler mit Bild: {image_path}")

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Embeddings saved in {metadata_file} ✅")
    print(f'Total embeddings created: {len(metadata)}')
 
create_image_embeddings(model, "vase_teapot/test/00849", metadata_file="embeddings/embeddings_metadata_teapots_test.json", class_name=0)
create_image_embeddings(model, "vase_teapot/test/00883", metadata_file="embeddings/embeddings_metadata_vases_test.json", class_name=1)
