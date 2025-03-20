
import streamlit as st
import json
from PIL import Image
import torch
import torch.nn as nn
from concepts_func import get_concepts_relevance_range 
from concepts_func import get_concept
import torchvision.transforms as T


with open("../embeddings/embeddings_metadata_teapots_test.json", "r") as f:
   data = json.load(f)

with open("../embeddings/embeddings_metadata_teapots_test.json", "r") as f:
    metadata_teapots = json.load(f)

with open("../embeddings/embeddings_metadata_vases_test.json", "r") as f:
    metadata_vases = json.load(f)

image_files = [item["image_path"] for item in data]



if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "input_image_path" not in st.session_state:
    st.session_state.input_image_path = ""

if "best_match_image" not in st.session_state:
    st.session_state.best_match_image = None

if "highest_similarity" not in st.session_state:
    st.session_state.highest_similarity = -1

if 'concept_inputs' not in st.session_state:
    st.session_state.concept_inputs = []

if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

columns_per_row =4
if 'concept_inputs_contr' not in st.session_state:
    st.session_state.concept_inputs_contr = []


st.title("🖼️ Select an Image")
st.markdown("Select an image and name the concepts to generate concept-based contrastive explanation 😌")

with st.expander("📂 Click to Show Image List", expanded=st.session_state.selected_image is None):
    cols = st.columns(5)

    for i, image_path in enumerate(image_files):
        image = Image.open(image_path)
        with cols[i % 5]:
            if st.button(f"Select Image {i}", key=f"select_{i}"):
                st.session_state.selected_image = image_path
                st.session_state.input_image_path = image_path  
                st.session_state.highest_similarity = -1
                st.session_state.concept_inputs = []
                st.session_state.concept_inputs_contr = []
                st.session_state.confirmed = False
            st.image(image, use_container_width=True)


if st.session_state.selected_image:
    st.write("### Selected Image")
    selected_image = Image.open(st.session_state.selected_image)



if st.session_state.input_image_path:
    input_embedding = None
    for entry in metadata_teapots:
        if entry["image_path"] == st.session_state.input_image_path:
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

        
        if best_match:
            st.session_state.best_match_image = best_match["image_path"]
            st.session_state.highest_similarity = highest_similarity

col1, col2 = st.columns(2)

# Display most similar image
if st.session_state.best_match_image:
    st.write(f"**Cosine Similarity:** {st.session_state.highest_similarity:.4f}")
    cosine_image = Image.open(st.session_state.best_match_image)
    with col1:
        st.image(selected_image, caption="Teapot", use_container_width=True, width=500)
    with col2:
        st.image(cosine_image, caption="Vase", use_container_width=True, width=500)


device = "cuda:0"
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

if st.session_state.input_image_path:
    image = Image.open(st.session_state.input_image_path)
    sample = transform(image).unsqueeze(0).to(device)
    concepts, relevance = get_concepts_relevance_range(sample, 0, 0.25, class_index=0)
    st.write(f"The most relevant concepts of teapot: {concepts.tolist()}")


if st.session_state.best_match_image:
    image_match = Image.open(st.session_state.best_match_image)
    sample_match = transform(image_match).unsqueeze(0).to(device)
    concepts_match, relevance_match = get_concepts_relevance_range(sample_match, 0, 0.25, class_index=1)
    st.write(f"The most relevant concepts of vase: {concepts_match.tolist()}")


columns_per_row =4
temp_inputs = []


for i in range(0, len(concepts), columns_per_row):

    cols = st.columns(columns_per_row)
    for j in range(columns_per_row):

        if i+j < len(concepts):
            concept_idx = i+j
            concept_name = concepts[concept_idx]

            with cols[j]:
                st.image(get_concept(concepts, st.session_state.input_image_path, "0", concept_index=i+j))

                disable_input = st.checkbox(f"Undefined concept", key=f"disable_{concept_idx}")

                if disable_input:
                    input_value = concept_name.item()
                else:
                    input_value = st.text_input(
                        f"Insert concept name:",
                        key=f"input_{concept_idx}"
                    )
                temp_inputs.append(input_value)




temp_contr_inputs = []

for i in range(0, len(concepts_match), columns_per_row):

    cols = st.columns(columns_per_row)
    for j in range(columns_per_row):

        if i+j < len(concepts_match):
            concept_idx = i+j
            concept_name = concepts_match[concept_idx]

            with cols[j]:
                st.image(get_concept(concepts_match, st.session_state.best_match_image, "1", concept_index=i+j))
                
                disable_input = st.checkbox(f"Undefined concept", key=f"disable_contr_{concept_idx}")

                if disable_input:
                    input_value = concept_name.item()
                else:
                    input_value = st.text_input(
                        f"Insert concept name:",
                        key=f"input_contr_{concept_idx}"
                    )
                temp_contr_inputs.append(input_value)
                


if st.button("Confirm"):
    st.session_state.concept_inputs = temp_inputs
    st.session_state.concept_inputs_contr = temp_contr_inputs
    st.session_state.confirmed = True

if st.session_state.confirmed:
    st.write(f"The model classified the image as a Teapot instead of a Vase because it contains the concepts {', '.join([str(x) for x in st.session_state.concept_inputs])}, and does not contain the concepts {', '.join([str(x) for x in st.session_state.concept_inputs_contr])}.")




