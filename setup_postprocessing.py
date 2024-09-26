import torch 
torch.cuda.set_per_process_memory_fraction(0.7)

class PostprocessingNetwork(torch.nn.Module):
    def __init__(self, hidden_channels = 16, kernel_size = 3):
        super(PostprocessingNetwork, self).__init__()

        self.conv1 = torch.nn.Conv3d(1, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv2 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv3 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

        self.max_pool = torch.nn.MaxPool3d(kernel_size=2)

        self.conv4 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv5 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

        self.conv6 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv7 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

        self.conv8 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv9 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

        # interpolate 

        self.conv10 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv11 = torch.nn.Conv3d(hidden_channels, 1, kernel_size, padding='same', bias=False)

        self.activation = torch.nn.ReLU()

        #self.list_of_conv3[-1].weight.data.fill_(0.0)
        #self.list_of_conv3[-1].bias.data.fill_(0.0)

    def forward(self, x):

        shape = x.shape
        z = self.activation(self.conv1(x))
        z = self.activation(self.conv2(z))
        z1 = self.activation(self.conv3(z))

        z2 = self.max_pool(z1) # shape // 2
        z2 = self.activation(self.conv4(z2))
        z2 = self.activation(self.conv5(z2))

        z3 = self.max_pool(z2) # shape // 4
        z2 = self.activation(self.conv6(z2))
        z2 = self.activation(self.conv7(z2))

        upsampling_shape = shape[-3:]
        upsampling_shape = [s // 2 for s in upsampling_shape]
        z4 = torch.nn.functional.interpolate(z2, size=upsampling_shape, mode = "trilinear", align_corners=True)
        z4 = z4 + z2 

        z4 = self.activation(self.conv8(z4))
        z4 = self.activation(self.conv9(z4))

        z5 = torch.nn.functional.interpolate(z2, size=shape[-3:], mode = "trilinear", align_corners=True)
        z5 = z5 + z1 

        z6 = self.activation(self.conv10(z5))
        z_out =self.conv11(z6)

        return z_out

DEVICE = "cuda"

postprocessing_model = PostprocessingNetwork()
postprocessing_model = postprocessing_model.to(DEVICE)
postprocessing_model.eval() 
postprocessing_model.load_state_dict(torch.load("checkpoint/postprocessing_model.pt", weights_only=True))

