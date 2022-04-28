from torch.nn import functional as F
import torchvision.models as models
from .basic import BasicModule,ChannelAttention
import torch.nn as nn
from functools import partial
import numpy as np
import torch as t
import math
from .basic import ResnetBlock
from .basic import CrissCrossAttention
from .basic import BasicBlock1DConv
from .basic import DBlock,DBlock_ca
from .basic import DecoderBlock
from .basic import DecoderBlock1DConv2
from .basic import DecoderBlock1DConv4
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
nonlinearity = partial(F.relu, inplace=True)
def get_upsampling_weight(input_channel, output_channel, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size,:kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((input_channel, output_channel, kernel_size, kernel_size))
    for i in range(input_channel):    
        weight[i,range(output_channel),:,:] = filt
    return t.from_numpy(weight).float()

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = get_upsampling_weight(
                m.in_channels, 
                m.out_channels, 
                m.kernel_size[0]
            )
            m.weight.data.copy_(initial_weight)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return F.leaky_relu(out)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, dilation=dilation, padding=dilation, bias=False),
        self.bn2 = nn.BatchNorm2d(planes),
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False),
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return F.leaky_relu(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = t.mean(x, dim=1, keepdim=True)
        max_out, _ = t.max(x, dim=1, keepdim=True)
        x = t.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResNet(BasicModule):
    def __init__(self, name, block, layers, input_channel, output_channel, pretrained=False):
        super(ResNet, self).__init__()
        self.name = name
        self.inplanes = 16
        self.conv1 = nn.Conv2d(input_channel, 16, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=3, dilation=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=1, dilation=4)
        self.final1=nn.Conv2d(128,2,1,1,bias=False)
        self.final2=nn.AdaptiveAvgPool2d(1)
        initialize_weights(self)
        if pretrained:
            print('load the pretrained model...')
            pretrained_model = '../pretrained_model/resnet34-333f7ec4.pth'
            resnet = models.resnet34()
            resnet.load_state_dict(t.load(pretrained_model))
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final1(x)
        x = self.final2(x)
        x = t.sigmoid(x)
        x=x.squeeze(-1)
        x=x.squeeze(-1)
        x=x.squeeze(-1)
        return x

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(t.cat([x, output], 1))
        return output

class DinkNet34_no(BasicModule):
    def __init__(self,name, num_classes=2, num_channels=3, encoder_1dconv=1, decoder_1dconv=0):
        super(DinkNet34_no, self).__init__()
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        self.name=name
        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        if encoder_1dconv == 0:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4
        else:
            myresnet = ResnetBlock()
            layers = [3, 4, 6, 3]
            basicBlock = BasicBlock1DConv
            self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
            self.encoder2 = myresnet._make_layer(
                basicBlock, 128, layers[1], stride=2)
            self.encoder3 = myresnet._make_layer(
                basicBlock, 256, layers[2], stride=2)
            self.encoder4 = myresnet._make_layer(
                basicBlock, 512, layers[3], stride=2)
        self.dblock = DBlock(256)
        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
            self.decoder = DecoderBlock1DConv4
        self.decoder4 = self.decoder(filters[3], filters[2])
        self.decoder3 = self.decoder(filters[2], filters[1])
        self.decoder2 = self.decoder(filters[1], filters[0])
        self.decoder1 = self.decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)

        if self.num_channels > 3:
            self.addconv = nn.Conv2d(
                self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)
    def forward(self, x):
        if self.num_channels > 3:
            add = self.addconv(x.narrow(1, 3, self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))
            x = x + add
        else:
            x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e3 = self.dblock(e3)
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finalconv3(d1)

        return t.sigmoid(out)

class DinkNet34_arbiter(BasicModule):
    def __init__(self,name, num_classes=2, num_channels=3, encoder_1dconv=1, decoder_1dconv=0):
        super(DinkNet34_arbiter, self).__init__()
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        self.name=name
        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        if encoder_1dconv == 0:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4
        else:
            myresnet = ResnetBlock()
            layers = [3, 4, 6, 3]
            basicBlock = BasicBlock1DConv
            self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
            self.encoder2 = myresnet._make_layer(
                basicBlock, 128, layers[1], stride=2)
            self.encoder3 = myresnet._make_layer(
                basicBlock, 256, layers[2], stride=2)
            self.encoder4 = myresnet._make_layer(
                basicBlock, 512, layers[3], stride=2)
        self.dblock = DBlock(256)
        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
            self.decoder = DecoderBlock1DConv4
        self.decoder4 = self.decoder(filters[3], filters[2])
        self.decoder3 = self.decoder(filters[2], filters[1])
        self.decoder2 = self.decoder(filters[1], filters[0])
        self.decoder1 = self.decoder(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.finalconv4 = nn.AdaptiveAvgPool2d(1)

        if self.num_channels > 3:
            self.addconv = nn.Conv2d(
                self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)
    def forward(self, x):
        if self.num_channels > 3:
            add = self.addconv(x.narrow(1, 3, self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))
            x = x + add
        else:
            x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e3 = self.dblock(e3)
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finalconv3(d1)
        out = self.finalconv4(out)
        return t.sigmoid(out)
class ResNet_ari(BasicModule):
    def __init__(self, name, block, layers, input_channel, output_channel, pretrained=False):
        super(ResNet_ari, self).__init__()
        self.name = name
        self.inplanes = 3
        self.sigmoid=t.nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_channel, 3, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 3, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 6, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 12, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 24, layers[3], stride=1, dilation=1)
        self.final1=nn.Conv2d(24,output_channel,1,1,bias=False)
        self.final2=nn.AdaptiveAvgPool2d(1)
        initialize_weights(self)
        if pretrained:
            print('load the pretrained model...')
            pretrained_model = '../pretrained_model/resnet34-333f7ec4.pth'
            resnet = models.resnet34()
            resnet.load_state_dict(t.load(pretrained_model))
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final1(x)
        x = self.final2(x)
        x=x.view(x.size(0),-1)
        x=self.sigmoid(x)
        return x

class DinkNet34_double(BasicModule):
    def __init__(self,name, num_classes=2, num_channels=6, encoder_1dconv=1, decoder_1dconv=0):
        super(DinkNet34_double, self).__init__()
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        self.name=name
        self.cc = RCCAModule(64, 64, 512)
        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        if encoder_1dconv == 0:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4
        else:
            myresnet = ResnetBlock()
            layers = [3, 4, 6, 3]
            basicBlock = BasicBlock1DConv
            self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
            self.encoder2 = myresnet._make_layer(
                basicBlock, 128, layers[1], stride=2)
            self.encoder3 = myresnet._make_layer(
                basicBlock, 256, layers[2], stride=2)
            self.encoder4 = myresnet._make_layer(
                basicBlock, 512, layers[3], stride=2)
        self.dblock = DBlock(256)
        self.ca1=ChannelAttention(256)
        self.sa1=SpatialAttention()
        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
            self.decoder = DecoderBlock1DConv4
        self.decoder4 = self.decoder(filters[3], filters[2])
        self.decoder3 = self.decoder(filters[2], filters[1])
        self.decoder2 = self.decoder(filters[1], filters[0])
        self.decoder1 = self.decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)

        self.dou_decoder3 = self.decoder(filters[2], filters[1])
        self.dou_decoder2 = self.decoder(filters[1], filters[0])
        self.dou_decoder1 = self.decoder(filters[0], filters[0])
        self.dou_finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1)


        if self.num_channels > 3:
            self.addconv = nn.Conv2d(
                self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)
    def forward(self, x):
        if self.num_channels > 3:
            add = self.addconv(x.narrow(1, 3, self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))

            x = x + add
        else:
            x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e3 = self.dblock(e3)
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finalconv3(d1)

        dou_d3=self.dou_decoder3(e3)+e2
        dou_d2=self.dou_decoder2(dou_d3)+e1
        dou_d1=self.dou_decoder1(dou_d2)
        dou_out=self.dou_finalconv3(dou_d1)

        out_junc=t.sigmoid((out))
        out_wall=t.sigmoid((dou_out))
        out_all= t.cat([out_junc, out_wall], dim=1)


        return out_all

def model(module_name, model_name, **kwargs):
    model_type = model_name.split('_')[0]
    name = module_name + "_" + model_type
    if "resnet18" in model_type:
        return ResNet(name, BasicBlock, [2, 2, 2, 2], **kwargs)
    if "resnet34" in model_type:
        return ResNet(name, BasicBlock, [3, 4, 6, 3], **kwargs)
    if "resnet50" in model_type:
        return ResNet(name, Bottleneck, [3, 4, 6, 3], **kwargs)
    if "resnet101" in model_type:
        return ResNet(name, Bottleneck, [3, 4, 23, 3], **kwargs)
    if "resnet152" in model_type:  
        return ResNet(name, Bottleneck, [3, 8, 36, 3], **kwargs)
    if "dlink34no" in model_type:
        return DinkNet34_no(name,**kwargs)
    if "dlink34double" in model_type:
        return DinkNet34_double(name,**kwargs)

    if "res18ari" in model_type:
        return ResNet_ari(name, BasicBlock, [1, 1, 1, 1],**kwargs)