import torch
from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = vgg16_bn(True).to(device)
model.eval()

canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonPlusFlat(canonizers)

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

image = Image.open("zennit-crp_new/tutorials/images/pelican.JPEG")

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

imgify(attr.heatmap, symmetric=True)