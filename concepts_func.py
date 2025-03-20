import torchvision
import torch
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization
from model_freezed import VGG16_piano
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from crp.helper import get_layer_names
import os
import shutil
from crp.image import vis_opaque_img
from crp.image import plot_grid

device = "cuda:0" if torch.cuda.is_available() else "cpu"

cc = ChannelConcept()
model = VGG16_piano().to(device)
model.load_state_dict(torch.load("models/teapot_vase_f1score_maro_vgg16.pth", map_location=torch.device('cpu')))
model.eval()
layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
layer_map = {layer : cc for layer in layer_names}


attribution = CondAttribution(model)

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


train_dataset = datasets.ImageFolder(root="vase_teapot/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


fv_path = "analyse/Vase_Teapot_ImageNet_analyse_bachelor"
fv_full = FeatureVisualization(attribution, train_dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)


canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonPlusFlat(canonizers)

def get_concepts_relevance_range(sample, start_range, end_range, class_index):
  
  sample.requires_grad = True

  conditions = [{'y': [class_index]}]
  attr = attribution(sample, conditions, composite, record_layer=layer_names)

  rel_c = cc.attribute(attr.relevances['vgg.features.40'], abs_norm=True)

  rel_values = rel_c[0] *100
  rel_values = torch.clamp(rel_values, min=0)

  sorted_values, sorted_indices = torch.sort(rel_values, descending=True)

  cumsum_rel = torch.cumsum(sorted_values, dim=0)
  total_rel = cumsum_rel[-1]

  # Compute cumulative relevance thresholds
  lower_thresh = start_range * total_rel
  upper_thresh = end_range * total_rel

  # Find concept indices that fall within the cumulative range
  in_range_mask = (cumsum_rel >= lower_thresh) & (cumsum_rel <= upper_thresh)
  selected_indices = sorted_indices[in_range_mask]
  selected_values = sorted_values[in_range_mask]

  return selected_indices, selected_values




def create_dataset_with_image(image_path: str, class_name: str, base_dir="custom_dataset"):

    # Find the next available custom_dataset{i} folder
    i = 1
    while os.path.exists(f"instanceDataset/{base_dir}{i}"):
        i += 1
    
    
    if i > 100:
        for j in range(1, i):
            folder_to_delete = f"instanceDataset/{base_dir}{j}"
            if os.path.exists(folder_to_delete):
                shutil.rmtree(folder_to_delete) 
                print(f"Deleted folder: {folder_to_delete}")
        i = 1

    dataset_path = f"instanceDataset/{base_dir}{i}"

    class_path = os.path.join(dataset_path, class_name)
    os.makedirs(class_path, exist_ok=True)


    shutil.copy(image_path, class_path)
    print(dataset_path)
    return dataset_path 



def get_concept(concepts, image_path, class_name, concept_index):
  '''
  Returns concept and its figure
  '''

  concept_ids = concepts.tolist()

  dataset_path = create_dataset_with_image(image_path, class_name)
  teapot_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
  fv_teapot = FeatureVisualization(attribution, teapot_dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

  ref_c = fv_teapot.get_max_reference(concept_ids[concept_index], "vgg.features.40", "relevance", (0, 8), composite=composite, plot_fn=vis_opaque_img)
  return plot_grid(ref_c, figsize=(6, 9), padding=False)


def generate_contrastive_explanation(concepts_array1, concepts_array2):
   concepts_array1 = concepts_array1.tolist()
   concepts_array2 = concepts_array2.tolist()
   concepts_array1 = set(concepts_array1) 
   concepts_array2 = set(concepts_array2)

   contrastive_img1 = concepts_array1 - concepts_array2
   contrastive_img2 = concepts_array2 - concepts_array1

   return contrastive_img1, contrastive_img2




