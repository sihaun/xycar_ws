import torchvision
import torch
from torch2trt import torch2trt

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_model_xy.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.cuda().eval().half()

data = torch.randn((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)

output_trt = model_trt(data)
output = model(data)

print(output.flatten()[0:10])
print(output_trt.flatten()[0:10])
print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))

torch.save(model_trt.state_dict(), 'best_model_xy_trt.pth')
