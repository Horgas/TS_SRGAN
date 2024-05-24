# plotting the trajectories (1D and 3D), loss



# converting the list to tensors and changing dimensions
g_loss_tensor = torch.tensor(g_loss_tensor)
g_MSEloss_tensor = torch.tensor(g_MSEloss_tensor)
print(g_loss_tensor.shape)
new_dim_gLoss = g_loss_tensor.view(-1,20).mean(dim=1) #reshape, mean for every batch
new_dim_mseLoss = g_MSEloss_tensor.view(-1,20).mean(dim=1)
print(new_dim_gLoss.shape)

def plot_GeneratorLoss()
    loss_figure = plt.figure()
    
    axes = plt.axes()
    log_g_loss = torch.log(new_dim_gLoss)
    log_MSEloss = torch.log(new_dim_mseLoss)
    
    
    axes.plot(log_g_loss, label='Generator Total Loss', color='purple')
    axes.plot(log_MSEloss,label='Generator MSE Loss', color='green')
    
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('log Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_Disc_AdvLoss() 
    loss_figure = plt.figure()
    
    axes = plt.axes()
    axes.plot(g_ADVloss_tensor, label='Generator Adversarial Loss', color='red')
    axes.plot(d_loss_tensor,label='Discriminator Loss', color='blue')
    plt.title('Training Losses ')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

#plot for a sample super-resolution trajectory
def sr_trajectory()
    sample_test_lr = test_dataset[10]['lr'].unsqueeze(0)
    with torch.no_grad():
        generated_data = generator(sample_test_lr.to(device))
    
    generated_data_np = generated_data.detach().cpu().numpy()
    #print(generated_data_np)
    x_sr = generated_data_np[0,0,:]
    y_sr = generated_data_np[0,1,:]
    z_sr = generated_data_np[0,2,:]
    
    a = 5
    fig = plt.figure(facecolor='w', figsize=(6.4*a, 4.8*a)) #creates figure
    
    axes = plt.axes(projection="3d") # 3d plot area for 3d data
    axes.set_facecolor('w')
    axes.plot(x_sr,y_sr,z_sr,'black', linewidth=0.8)
    axes.scatter3D(x_sr, y_sr, z_sr, c=z_sr, cmap='ocean', s = 8)
    
    plt.grid(False)
    plt.axis('off')
    plt.title('Lorenz generated',color='white', fontsize=10*a)
    plt.show()


#plot for a sample low-resolution trajectory
def lr_trajectory()
    sample_test_lr = test_dataset[10]['lr']
    #print(sample_test_lr)
    sample_test_lr_np = sample_test_lr.detach().cpu().numpy()
    x_lr = sample_test_lr_np[0,:]
    y_lr = sample_test_lr_np[1,:]
    z_lr = sample_test_lr_np[2,:]
    
    a = 5
    fig = plt.figure(facecolor='w', figsize=(6.4*a, 4.8*a)) #creates figure
    
    axes = plt.axes(projection="3d") # 3d plot area for 3d data
    axes.set_facecolor('w')
    axes.plot(x_lr,y_lr,z_lr,'black', linewidth=0.8)
    axes.scatter3D(x_lr, y_lr, z_lr, c=z_lr, cmap='ocean', s = 8)
    
    plt.grid(False)
    plt.axis('off')
    plt.title('Lorenz low resolution',color='white', fontsize=10*a)
    plt.show()


#plot for a sample high-resolution trajectory
def hr_trajectory()
    sample_test_hr = test_dataset[10]['hr']
    #print(sample_test_lr)
    sample_test_hr_np = sample_test_hr.detach().cpu().numpy()
    x_hr = sample_test_hr_np[0,:]
    y_hr = sample_test_hr_np[1,:]
    z_hr = sample_test_hr_np[2,:]
    
    a = 5
    fig = plt.figure(facecolor='w', figsize=(6.4*a, 4.8*a)) #creates figure
    
    axes = plt.axes(projection="3d") # 3d plot area for 3d data
    axes.set_facecolor('w')
    axes.plot(x_hr,y_hr,z_hr,'black', linewidth=0.8)
    axes.scatter3D(x_hr, y_hr, z_hr, c=z_hr, cmap='ocean', s = 8)
    
    plt.grid(False)
    plt.axis('off')
    plt.title('Lorenz high resolution',color='white', fontsize=10*a)
    plt.show()

''' Calculation of MSE and relative distance
For temporal upscaling the calulations between the low resolution and high resolution data needs to be commented out
because they have a different amount of timesteps '''
mse_values = []
mse_values_lr_hr = []
relative_difference_values = []
relative_difference_values2 = []
abs_diff_values = []
abs_diff_values2 = []

def get_mse()
    for batch in test_loader:
        lr_test = batch['lr']  # Low resolution  data
        hr_test = batch['hr']  # Corresponding high resolution data
        
        hr_test_tensor = torch.tensor(hr_test,dtype=torch.float32)
        lr_test_tensor = torch.tensor(lr_test,dtype=torch.float32)
        # Generate sr
        with torch.no_grad():
            sr_test = generator(lr_test.to(device))
    
    
    
        sr_test_tensor = sr_test.detach().cpu()    # print(len(test_dataset))
        # print(sr_test_tensor.shape)
    
        # Calculates MSE for each pair of SR and HR in  batch
        for sr, hr in zip(sr_test_tensor, hr_test_tensor):
            #print(hr.shape)
            mse = loss_function_mse(sr, hr)
            mse_values.append(mse.item())
          
        #'''
        # Calculates MSE for each pair of LR and HR in  batch
        for lr, hr in zip(lr_test_tensor, hr_test_tensor):
            #print(hr.shape)
            mse2 = loss_function_mse(lr,hr)
            mse_values_lr_hr.append(mse2.item())
        #'''
        return mse_values, mse_values_lr_hr
  
    max_value = torch.max(hr_test_tensor).item()

  def get_relativeDistance()
      # Calculates relative distance for each pair of SR and HR in  batch
      for sr, hr in zip(sr_test_tensor, hr_test_tensor):
            #print(hr.shape)
            abs_diff = loss_L1(sr, hr)
            abs_diff_values.append(abs_diff.item())
            relative_difference = abs_diff.item() / max_value
            relative_difference_values.append(relative_difference)
      #'''
      # Calculates relative distance for each pair of LR and HR in  batch
      for lr, hr in zip(lr_test_tensor, hr_test_tensor):
          #print(hr.shape)
          abs_diff2 = loss_L1(lr, hr)
          abs_diff_values2.append(abs_diff2.item())
          relative_difference2 = abs_diff2.item() / max_value
          relative_difference_values2.append(relative_difference2)
      #'''
      return relative_difference_values, relative_difference_values2
#print(mse_values)
mse_sr_hr, mse_lr_hr = get_mse()
rel_diff_sr, rel_diff_lr = get_relativeDistance()
mse_average = sum(mse_sr_hr) / len(mse_sr_lr)
mse2_average = sum(mse_lr_hr) / len(mse_lr_hr)
abs_diff_avg_sr = sum(abs_diff_values) / len(abs_diff_values)
abs_diff_avg_lr = sum(abs_diff_values2) / len(abs_diff_values2)
# print(f"Average MSE sr,hr: {mse_average}")
# print(f"Average MSE lr,hr: {mse2_average}")
diff_average_sr = sum(rel_diff_sr) / len(rel_diff_sr)
diff_average_lr = sum(rel_diff_lr) / len(rel_diff_lr)
# print(" relative sr:", diff_average_sr)
# print("relative lr:", diff_average_lr)
# print("abs difference sr: ", abs_diff_avg_sr)
# print("abs difference lr: ", abs_diff_avg_lr)


def plot_relative_distance()
    plt.title("relative distance")
    plt.xlabel("test set")
    plt.ylabel("relative distance")
    plt.yscale("log")
    #plt.plot(mse_values, label="hr-sr")
    #plt.plot(mse_values_lr_hr, label="hr-lr")
    plt.plot(rel_diff_sr, label ="Relative difference HR-SR")
    plt.plot(rel_diff_lr, label = "Relative difference HR-LR")
    plt.legend()
    plt.show()
#1D representation

def plot_1D()
    plt.figure(figsize=(9, 4))
    
    plt.xlabel("time")
    plt.ylabel("x")
    
    time = np.linspace(0, 7, 800)
    plt.plot(time,x_lr,  'cyan', label='Low Resolution')
    plt.plot(time,x_hr, 'magenta', label='High Resolution')
    plt.plot(time,x_sr, 'green', label='Super Resolution')
    plt.legend()
    #plt.gca().set_facecolor('white')
    plt.show()
    plt.figure(figsize=(9, 4))
    
    plt.xlabel("time")
    plt.ylabel("y")
    
    
    plt.plot(time,y_lr, 'cyan', label='Low Resolution')
    plt.plot(time,y_hr, 'magenta', label='High Resolution')
    plt.plot(time,y_sr, 'green', label='Super Resolution')
    plt.legend()
    plt.show()
    plt.figure(figsize=(9, 4))
    
    plt.xlabel("time")
    plt.ylabel("z")
    
    
    plt.plot(time,z_lr, 'cyan', label='Low Resolution')
    plt.plot(time,z_hr, 'magenta', label='High Resolution')
    plt.plot(time,z_sr, 'green', label='Super Resolution')
    plt.legend()
    plt.show()
