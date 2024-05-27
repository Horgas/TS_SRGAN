import torch
import torch.optim as optim
import numpy as np
from scipy.integrate import solve_ivp

#training

channels = 3
num_channels = 3
length_hr = high_res_data[1].shape
sequence_length = amount_val
a = 0.1
num_epochs = 1000 # for temporal upscaling a lower epochs number is sufficient
upscale_factor = ds_factor

g_MSEloss_tensor = []
g_ADVloss_tensor = []
d_loss_tensor = []
g_loss_tensor = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator_net(upscale_factor, num_blocks = 16,).to(device) # num_blocks defines the depth of the network
discriminator = Discriminator(input_shape=(channels,*length_hr)).to(device)


optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))



mock_input = torch.randn(1, num_channels, sequence_length).to(device)  # Adjust num_channels and sequence_length

with torch.no_grad():
    mock_output = discriminator(mock_input)

# the output shape
discriminator_output_shape = mock_output.shape[1:]


for epoch in range(num_epochs):
    print("num epoch: ", epoch)
    for i,batch in enumerate(train_loader):

        discriminator.train()
        generator.train()
        #print("batch lr: ", batch['lr'].shape)

        low_res_data, high_res_data = batch['lr'].to(device), batch['hr'].to(device)

        valid = torch.ones((low_res_data.size(0), *discriminator_output_shape), device=device)
        fake = torch.zeros((low_res_data.size(0), *discriminator_output_shape), device=device)
        # Generate data
        #print(low_res_data.shape)
        generated_data = generator(low_res_data)
        #print(generated_data.shape)


        optimizer_D.zero_grad()

        loss_real = loss_function_adv(discriminator(high_res_data), valid) #tries to label real as real
        loss_fake = loss_function_adv(discriminator(generated_data.detach()), fake) #tries to label fake as fake

        output_D_real = discriminator(high_res_data)
        output_D_fake = discriminator(generated_data.detach())
        # print("r: ",output_D_real)
        # print("f: ",output_D_fake)

        loss_D = (loss_real + loss_fake) / 2
        d_loss_tensor.append(loss_D.item())

        #Backpropagation Discriminator

        loss_D.backward()
        optimizer_D.step()




        # Train Generator

        optimizer_G.zero_grad()

        loss_gan = loss_function_adv(discriminator(generated_data), valid).to(device) #tries to fool the D. the fake is real
        g_ADVloss_tensor.append(loss_gan.item())
        loss_mse = loss_function_mse(generated_data, high_res_data).to(device)
        g_MSEloss_tensor.append(loss_mse.item())

        loss_G = loss_mse + a * loss_gan
        g_loss_tensor.append(loss_G.item())


        #Backpropagation Generator
        loss_G.backward()
        optimizer_G.step()
