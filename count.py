import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image

#####
from network.mynetwork import Unet
from network.mynetwork_cmp import Unet as Unet_cmp
from network.styler import Unet as Unet_styler
from loss.loss import CLIPLoss
######





from thop import profile

device = "cuda" if torch.cuda.is_available() else "cpu"

input_pic = torch.ones(1, 3, 512, 512).to(device)
input1 = torch.ones(1, 3, 244, 244).to(device).long()
input2 = torch.ones(1, 77).to(device).long()


print("clip")
model = CLIPLoss(device).to(device)
model = model.model
clip_flops, clip_params = profile(model, inputs=(input1, input2), verbose=False)
print('FLOPs = ' + str(clip_flops/1000**3) + 'G')
print('Params = ' + str(clip_params/1000**2) + 'M')

print()

print("vgg19")
vgg = torchvision.models.vgg19(pretrained=True).features.to(device)
vgg_flops, vgg_params = profile(vgg, inputs=(input_pic,), verbose=False)
print('FLOPs = ' + str(vgg_flops/1000**3) + 'G')
print('Params = ' + str(vgg_params/1000**2) + 'M')

print()


print("mine")
model_mine = Unet(device).to(device)
mine_flops, mine_params = profile(model_mine, inputs=(input_pic,), verbose=False)
print('FLOPs = ' + str((mine_flops+clip_flops)/1000**3) + 'G')
print('Params = ' + str((mine_params+clip_params)/1000**2) + 'M')

print()

print("cmp")
model = Unet_cmp(device).to(device)
cmp_flops, cmp_params = profile(model, inputs=(input_pic,), verbose=False)
print('FLOPs = ' + str((cmp_flops+clip_flops+vgg_flops)/1000**3) + 'G')
print('Params = ' + str((cmp_params+clip_params+vgg_params)/1000**2) + 'M')

print()

print("styler")
model = Unet_styler().to(device)
sty_flops, sty_params = profile(model, inputs=(input_pic,), verbose=False)
print('FLOPs = ' + str((sty_flops+clip_flops+vgg_flops)/1000**3) + 'G')
print('Params = ' + str((sty_params+clip_params+vgg_params)/1000**2) + 'M')


print()



