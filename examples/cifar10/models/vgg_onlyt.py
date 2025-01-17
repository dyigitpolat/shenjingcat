
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import catSNN
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M'],
    'o' : [128,128,'M',256,256,'M',512,512,'M',(1024,0),'M']

}


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

class VGG_o_dense(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=True):
        super(VGG_o_dense, self).__init__()
        self.clamp_max = clamp_max
        self.bias = bias
        
        self.features1 = self._make_layers(3,128)
        self.features2 = self._make_layers(128,256)
        self.features3 = self._make_layers(256,512)
        self.features4 = self._make_layers(512,1024)
        self.liner = nn.Linear(1024, 10, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        
        out = self.features1(x)
        out = out.view(out.size(0), -1)
        out = self.liner(out)
        return out

    def _make_layers(self, in_channel,out_channel):
        layers = []
        #in_channels = 3
        conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=self.bias)
        conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=self.bias)
        conv3 = torch.cat(conv1, conv2)
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2), nn.Dropout2d(0.15)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=self.bias),nn.BatchNorm2d(out_channels),Clamp(),nn.Dropout2d(0.15)]
                in_channels = out_channels

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
          
class VGG_o(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=True):
        super(VGG_o, self).__init__()
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(1024, 10, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2), nn.Dropout2d(0.2)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [catSNN.QuantizedConv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=self.bias),nn.BatchNorm2d(out_channels),catSNN.Clamp_q(max = self.clamp_max),nn.Dropout2d(0.2)]
                in_channels = out_channels

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class CatVGG_o(nn.Module):
    def __init__(self, vgg_name, T, is_noise=False, bias=True):
        super(CatVGG_o, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.is_noise = is_noise
        self.bias = bias

        self.features = self._make_layers(cfg[vgg_name], is_noise)
        self.classifier = self.snn.dense((1, 1, 1024), 10,bias = True)

    def forward(self, x):
        
        out = self.features(x)
        out = self.classifier(out)
        out = self.snn.sum_spikes(out) / self.T
        return out

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2),nn.Dropout2d(0)]
            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
                    in_channels = x
                else:
                    padding = x[1] if isinstance(x, tuple) else 1
                    out_channels = x[0] if isinstance(x, tuple) else x
                    layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                               self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
                    in_channels = out_channels
        return nn.Sequential(*layers)

