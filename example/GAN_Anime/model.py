import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        

        self.convt1=nn.ConvTranspose2d( 100, 512, 4, 1, 0, bias=False)
        self.btnt1=nn.BatchNorm2d(512)
        self.relut1=nn.ReLU(True)
            
            # ``512*4*4``
        self.convt2=nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False)
        self.btnt2=nn.BatchNorm2d(256)
        self.relut2=nn.ReLU(True)
            
            # ``256*8*8``
        self.convt3=nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False)
        self.btnt3=nn.BatchNorm2d(128)
        self.relut3=nn.ReLU(True)
            
            # ``128*16*16``
        self.convt4=nn.ConvTranspose2d(128,64, 4, 2, 1, bias=False)
        self.btnt4=nn.BatchNorm2d(64)
        self.relut4=nn.ReLU(True)
            
            # 64*32*32``
        self.convt5= nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False)
        self.tan=nn.Tanh()
        
     # 3*64*64``
    def forward(self, Input):
        output=self.convt1(Input)
        output=self.btnt1(output)
        output=self.relut1(output)
        
        output=self.convt2(output)
        output=self.btnt2(output)
        output=self.relut2(output)
        
        output=self.convt3(output)
        output=self.btnt3(output)
        output=self.relut3(output)
        
        output=self.convt4(output)
        output=self.btnt4(output)
        output=self.relut4(output)
        
        output=self.convt5(output)
        output=self.tan(output)
        
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
            # input is `3x 64 x 64``
        self.conv1=nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.relu1=nn.LeakyReLU(0.2, inplace=True)
            
            # ``64 x 32 x 32``
        self.conv2= nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.btn2=nn.BatchNorm2d(128)
        self.relu2=nn.LeakyReLU(0.2, inplace=True)
            
            # ``128 x 16 x 16``
        self.conv3= nn.Conv2d(128,256, 4, 2, 1, bias=False)
        self.btn3=nn.BatchNorm2d(256)
        self.relu3=nn.LeakyReLU(0.2, inplace=True)
            
            # ``256 x 8 x 8``
        self.conv4= nn.Conv2d(256,512, 4, 2, 1, bias=False)
        self.btn4=nn.BatchNorm2d(512)
        self.relu4=nn.LeakyReLU(0.2, inplace=True)
            
            # 512 x 4 x 4``
        self.conv5= nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.sig=nn.Sigmoid()
        

    def forward(self, Input):
        output=self.conv1(Input)
        output=self.relu1(output)
        
        output=self.conv2(output)
        output=self.btn2(output)
        output=self.relu2(output)
        
        output=self.conv3(output)
        output=self.btn3(output)
        output=self.relu3(output)
        
        output=self.conv4(output)
        output=self.btn4(output)
        output=self.relu4(output)
        
        output=self.conv5(output)
        output=self.sig(output)
        
        return output