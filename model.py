import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.down1 = downStep(1, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down4 = downStep(256, 512)
        self.conv1 = nn.Conv2d(512,1024,3)
        self.conv2 = nn.Conv2d(1024,1024,3)
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)
        self.tanh = nn.Tanh()
        

    def forward(self, x):
        # todo
        x, x_down_1 = self.down1.forward(x)
        x, x_down_2 = self.down2.forward(x)
        x, x_down_3 = self.down3.forward(x)
        x, x_down_4 = self.down4.forward(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up1.forward(x, x_down_4)
        x = self.up2.forward(x, x_down_3)
        x = self.up3.forward(x, x_down_2)
        x = self.up4.forward(x, x_down_1)
        x = self.final(x)
        x = self.tanh(x)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        # todo
        x = self.encoder(x)
        x_down_temp = x
        x = self.pool(x)
        return x, x_down_temp

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.upconv = nn.ConvTranspose2d(inC, outC, 2, 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

    def forward(self, x, x_down):
        # todo
        x = self.upconv(x)
        diff_x = torch.tensor([(x_down.size()[2]-x.size()[2])])
        diff_y = torch.tensor([(x_down.size()[3]-x.size()[3])])
        N,C,H,W = x_down.size()
        #x_down_corped = x_down[:,:,diff_x:(H - diff_x), diff_y:(W - diff_y)]
        x_down_corped = F.pad(x_down,[-diff_x//2, -(diff_x-diff_y//2),
                                      -diff_y//2, -(diff_y-diff_x//2)])
        x = torch.cat((x_down_corped, x), 1)
        #print(x.size())
        
        
        x = self.encoder(x)
        
    
        return x