import streamlit as st
import torch
from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import numpy as np

col1, col2 = st.columns(2)
col1.image("images/pelican.JPEG", caption="First image")
col2.image("images/p1.JPEG", caption="Second image")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = vgg16_bn(True).to(device)
model.eval()

canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonPlusFlat(canonizers)
image = Image.open("images/pelican.JPEG")
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
sample = transform(image).unsqueeze(0).to(device)

from crp.concepts import ChannelConcept

cc = ChannelConcept()
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
mask_map = {name: cc.mask for name in layer_names}
layer_map = {layer : cc for layer in layer_names}

attribution = CondAttribution(model)

# compute heatmap wrt. output 144 (pelican class)
conditions = [{"y": 144}]

# zennit requires gradients
sample.requires_grad = True
attr = attribution(sample, conditions, composite, mask_map=mask_map)
from crp.image import imgify

print(torch.equal(attr[0], attr.heatmap))



if st.button("Generate heatmap"):
  new_image = imgify(attr.heatmap, symmetric=True)

# Display with Streamlit
  st.image(new_image, caption="heatmap 1")
  #st.image(attr.heatmap)

attr = attribution(sample, conditions, composite, record_layer=layer_names)
# layer features.40 has 512 channel concepts
rel_c = cc.attribute(attr.relevances['features.40'], abs_norm=True)
# the six most relevant concepts and their contribution to final classification in percent
rel_values, concept_ids = torch.topk(rel_c[0], 6)
concept_ids, rel_values*100
#look at the heatmap of concepts
conditions = [{"features.40": [id], "y": 144} for id in concept_ids]
heatmap, _, _, _ = attribution(sample, conditions, composite)

concepts_heamap = imgify(heatmap, symmetric=True, grid=(1, len(concept_ids)))
if st.button("Generate concepts heatmap"):
  st.image(concepts_heamap)

import torchvision
from crp.visualization import FeatureVisualization

preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_path = "../../data/ImageNet_data"
imagenet_data = torchvision.datasets.ImageNet(data_path, transform=transform, split="val")  
fv_path = "../../data/VGG16_ImageNet"
fv = FeatureVisualization(attribution, imagenet_data, layer_map, preprocess_fn=preprocessing, path=fv_path)

from crp.image import plot_grid
from crp.image import vis_opaque_img

ref_c = fv.get_max_reference([288, 436,  71, 296, 386, 401], "features.40", "relevance",  (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)

concepts = plot_grid(ref_c, figsize=(6, 9), padding=False)

if st.button("Show concepts"):
  st.image(ref_c)

 