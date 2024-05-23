#generating data



def my_lorenz_system(t,vec,sigma,p,beta):
    x,y,z = vec
    dxdt = sigma*(y-x)
    dydt = x*(p-z)-y
    dzdt = x*y-beta*z
    return dxdt,dydt,dzdt


x0_center, y0_center, z0_center = 2, 3, -14
# number of timepoints in the trajectories:
amount_val = 800
#amount_val = 4000
# time interval:
t_span = (0,7) 
#t_span = (0,100)

def generating_sample_data(num_samples,ds_factor):
  data_hr = []
  data_lr = []
  radius = 0.1

  for _ in range(num_samples):

      #parameters
      sigma = 10
      beta = 8/3
      p = 28

      t_eval = np.linspace(t_span[0], t_span[1], amount_val) #array of time points for solution
#One of the next two section marked with ### need to be commented out for the other one to work
    #temporal upscaling:
    ###
      
      '''
      #p = np.random.uniform(10, 30)
      #x0,y0,z0 = np.random.rand(3) *5
      x0, y0, z0 = np.random.normal(loc=0, scale=10, size=3)
      solution = solve_ivp(my_lorenz_system, t_span = t_span, y0 = [x0,y0,z0], t_eval = t_eval, args = (sigma,p,beta))

      high_res_data = solution.y
      low_res_data = downsample_data(solution.y, ds_factor)
      '''      
    ###
    #changing accuracy:
    ###
      
      phi = np.random.uniform(0, 2 * np.pi)
      cos_theta = np.random.uniform(-1, 1)
      theta = np.arccos(cos_theta)
      r = radius  # Radius of the sphere for initial conditions

      x0 = x0_center + r * np.sin(theta) * np.cos(phi)
      y0 = y0_center + r * np.sin(theta) * np.sin(phi)
      z0 = z0_center + r * np.cos(theta)
       
      solution_hr = solve_ivp(my_lorenz_system, t_span = t_span, y0 = [x0,y0,z0], t_eval = t_eval, args = (sigma,p,beta),atol=1e-2, rtol=1e-2)
      solution_lr = solve_ivp(my_lorenz_system, t_span = t_span, y0 = [x0,y0,z0], t_eval = t_eval, args = (sigma,p,beta),atol=1e-1, rtol=1e-1)
      high_res_data = solution_hr.y
      low_res_data = downsample_data(solution_lr.y, ds_factor)
      
     ###   
        
        
      data_hr.append(high_res_data)
      data_lr.append(low_res_data)

  return np.array(data_hr), np.array(data_lr)


#low-resolution training data(temporal):

def downsample_data(data, downsample_factor):
    return data[:, ::downsample_factor]




#prepare the data for training:

num_samples =600
train_size = int(num_samples * 9/10)
#num_samples = 400
#train_size = int(num_samples * 4/5)
ds_factor = 1 # choose downsampling factor for temporal upscaling (2,4,...), for changing accuracy set to 1

high_res_data, low_res_data = generating_sample_data(num_samples,ds_factor)

train_hr, test_hr, train_lr, test_lr = train_test_split(high_res_data, low_res_data, train_size=train_size)

class LorenzDataset(Dataset):
    def __init__(self, high_res_data, low_res_data): # initialization
        self.high_res_data = high_res_data
        self.low_res_data = low_res_data

    def __len__(self):
        return len(self.high_res_data)  # returns the number of time series

    def __getitem__(self, idx):
        high_res_sample = torch.tensor(self.high_res_data[idx], dtype=torch.float32)
        low_res_sample = torch.tensor(self.low_res_data[idx], dtype=torch.float32)
        return {'hr': high_res_sample, 'lr': low_res_sample}


# dataset and dataloader
test_dataset = LorenzDataset(test_hr, test_lr)
dataset = LorenzDataset(train_hr, train_lr)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle = False)
train_loader = DataLoader(dataset, batch_size=20, shuffle = False)
