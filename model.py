#architecture

class Generator_net(nn.Module):
    def __init__(self, upscale_factor, num_blocks):
        super(Generator_net, self).__init__()

        
        self.conv1 = nn.Sequential(nn.Conv1d(3,64, kernel_size = 9, stride = 1, padding = 4),
                                        nn.PReLU())
        blocks = []
        for _ in range(num_blocks):
            blocks.append(SRResNet(64))

        self.blocks = nn.Sequential(*blocks)

        self.conv2 = nn.Sequential(nn.Conv1d(64,64, kernel_size = 3, stride = 1, padding = 1),
                                              nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(128,64, kernel_size = 3, stride = 1, padding = 1),
                                              nn.BatchNorm1d(64),nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(64,3 ,kernel_size = 9, stride = 1, padding = 4))
        dummy = math.log2(upscale_factor)
        k = int(dummy)
        
        dense_layer = []
        for _ in range(k):
            dense_layer += [
              nn.Conv1d(64, 128, 3, 1, 1),
              nn.BatchNorm1d(128),
              None,
              nn.PReLU(),

            ]

        self.dense_layer = nn.Sequential(*dense_layer)




    def pixelShuffle1D(self,input,r):
        #print("PS input shape: ", input.shape)
        batch_size, channel, length = input.shape
        channels_out = channel // r
        new_length = length * r
        input_view = input.view(batch_size, channels_out, r, length)
        output = input_view.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, channels_out, new_length)
        #print("PS output shape: ", output.shape)
        return output

    def forward(self, vec):
        x1 = self.conv1(vec)
        #print("After conv1 layer: ", x.shape)
        x = self.blocks(x1)
        #print("After block layer: ", x.shape)
        x2 = self.conv2(x)
        x = torch.add(x1,x2) #skip connection
        #print("After conv2 layer: ", x.shape)
        for i,layer in enumerate(self.dense_layer):
            if layer is None:
                x = self.pixelShuffle1D(x,r=2)  # Custom pixel shuffle operation
                #print(f"After dense layer {i}: ", x.shape)
            else:
                x = layer(x)
      
        output = self.conv4(x)
        #print("After conv3 layer: ", output.shape)
        return output



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        in_filters = 3  # initial number of filters
        layers = []
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(self.discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            #print(out_filters)
            in_filters = out_filters

        layers.append(nn.Conv1d(out_filters, 1, kernel_size=3, stride=1, padding = 1))

        self.model = nn.Sequential(*layers)

    def discriminator_block(self, in_filters, out_filters, first_block=False):
        layers = []
        layers.append(nn.Conv1d(in_filters, out_filters, kernel_size=3, stride=1, padding = 1))
        if not first_block:
            layers.append(nn.BatchNorm1d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv1d(out_filters, out_filters, kernel_size=3))
        layers.append(nn.BatchNorm1d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, data):
        output = self.model(data)
        return output

      


class SRResNet(nn.Module):
  def __init__(self, feature_num):
    super(SRResNet, self).__init__()
    self.convolutional_block = nn.Sequential(
        nn.Conv1d(feature_num, feature_num, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm1d(feature_num, 0.1),
        nn.PReLU(),
        nn.Conv1d(feature_num, feature_num, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm1d(feature_num, 0.1),
    )

  def forward(self,x):
      return self.convolutional_block(x) + x
