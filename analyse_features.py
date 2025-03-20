
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



device = "cuda:0" if torch.cuda.is_available() else "cpu"
cc = ChannelConcept()
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = VGG16_piano().to(device)
model.load_state_dict(torch.load(f"models/teapot_vase_f1score_maro_vgg16.pth"))
model.eval()

layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
layer_map = {layer : cc for layer in layer_names}

attribution = CondAttribution(model)

test_dataset = datasets.ImageFolder(root=f"vase_teapot/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

fv_path = f"analyse/Vase_Teapot_ImageNet_analyse_bachelor"
fv = FeatureVisualization(attribution, test_dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonPlusFlat(canonizers)

saved_files = fv.run(composite, 0, len(test_dataset), 32, 100)


