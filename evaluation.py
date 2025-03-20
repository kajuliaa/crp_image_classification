
import tqdm  
import json
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from concepts_func import get_concepts_relevance_range 
from concepts_func import generate_contrastive_explanation



device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

with open("embeddings/embeddings_metadata_teapots_test.json", "r") as f:
  metadata_teapots = json.load(f)

with open("embeddings/embeddings_metadata_vases_test.json", "r") as f:
  metadata_vases = json.load(f)

verystrong_relevance = []
strong_relevance =[]
low_relevance =[]
verylow_relevance = []

for item in tqdm(metadata_teapots, desc="Processed Images", unit="Images"):
    #Concepts for image
    image = Image.open(item['image_path'])
    sample = transform(image).unsqueeze(0).to(device)

    concepts_verystrong, relevance_verystrong = get_concepts_relevance_range(sample, 0, 0.25, class_index=0)
    concepts_strong, relevance_strong = get_concepts_relevance_range(sample, 0.25, 0.5, class_index=0)
    concepts_low, relevance_low = get_concepts_relevance_range(sample, 0.5, 0.75, class_index=0)
    concepts_verylow, relevance_verylow = get_concepts_relevance_range(sample, 0.75, 1, class_index=0)

    #Find most similar image
    input_embedding = None
    for entry in metadata_teapots:
        if entry["image_path"] == item['image_path']:
            input_embedding = entry["embedding"]
            break

    if input_embedding:
        best_match = None
        highest_similarity = -1
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for entry in metadata_vases:
            stored_embedding = torch.tensor(entry["embedding"]).unsqueeze(0).float()
            input_embedding_tensor = torch.tensor(input_embedding).unsqueeze(0).float()
            similarity = cos(input_embedding_tensor, stored_embedding).mean()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = entry




    #Concepts for second image
    image_similar = Image.open(best_match["image_path"])
    sample_similar = transform(image_similar).unsqueeze(0).to(device)

    concepts_similar_verystrong, relevance_similar_verystrong = get_concepts_relevance_range(sample_similar, 0, 0.25, class_index=1)
    concepts_similar_strong, relevance_similar_strong = get_concepts_relevance_range(sample_similar, 0.25, 0.5, class_index=1)
    concepts_similar_low, relevance_similar_low = get_concepts_relevance_range(sample_similar, 0.5, 0.75, class_index=1)
    concepts_similar_verylow, relevance_similar_verylow = get_concepts_relevance_range(sample_similar, 0.75, 1, class_index=1)

    c1_verystrong, c2_verystrong = generate_contrastive_explanation(concepts_verystrong, concepts_similar_verystrong)
    length_explanation_verystrong = len(c1_verystrong) + len(c2_verystrong)
    verystrong_relevance.append(length_explanation_verystrong)

    c1_strong, c2_strong = generate_contrastive_explanation(concepts_strong, concepts_similar_strong)
    length_explanation_strong = len(c1_strong) + len(c2_strong) 
    strong_relevance.append(length_explanation_strong)

    c1_low, c2_low = generate_contrastive_explanation(concepts_low, concepts_similar_low)
    length_explanation_low = len(c1_low) + len(c2_low)
    low_relevance.append(length_explanation_low)

    c1_verylow, c2_verylow = generate_contrastive_explanation(concepts_verylow, concepts_similar_verylow)
    length_explanation_verylow = len(c1_verylow) + len(c2_verylow)
    verylow_relevance.append(length_explanation_verylow)


with open("verystrong_relevance.json", "w") as f:
    json.dump(verystrong_relevance, f, indent=2)

with open("strong_relevance.json", "w") as f:
    json.dump(strong_relevance, f, indent=2)

with open("low_relevance.json", "w") as f:
    json.dump(low_relevance, f, indent=2)

with open("verylow_relevance.json", "w") as f:
    json.dump(verylow_relevance, f, indent=2)

