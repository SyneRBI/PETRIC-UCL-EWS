
import torch 


class NetworkPreconditioner(torch.nn.Module):
    def __init__(self, n_layers = 1, hidden_channels = 8, kernel_size = 3):
        super(NetworkPreconditioner, self).__init__()

        self.conv1 = torch.nn.Conv3d(3, 3*hidden_channels, kernel_size, groups=3, padding='same', bias=False)
        self.conv2 = torch.nn.Conv3d(3*hidden_channels, 3*hidden_channels, kernel_size, groups=3, padding='same', bias=False)
        self.conv3 = torch.nn.Conv3d(3*hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

        self.max_pool = torch.nn.MaxPool3d(kernel_size=2)

        self.conv4 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv5 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
    
        # interpolate 

        self.conv6 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv7 = torch.nn.Conv3d(hidden_channels, 1, kernel_size, padding='same', bias=False)

        self.activation = torch.nn.ReLU()

        #self.list_of_conv3[-1].weight.data.fill_(0.0)
        #self.list_of_conv3[-1].bias.data.fill_(0.0)

    def forward(self, x):
        shape = x.shape
        z = self.activation(self.conv1(x))
        z = self.activation(self.conv2(z))
        z1 = self.activation(self.conv3(z))

        z2 = self.max_pool(z1)
        z2 = self.activation(self.conv4(z2))
        z2 = self.activation(self.conv5(z2))

        z3 = torch.nn.functional.interpolate(z2, size=shape[-3:], mode = "trilinear", align_corners=True)
        z3 = z3 + z1 

        z4 = self.activation(self.conv6(z3))
        z_out = self.activation(self.conv7(z4))

        return z_out
