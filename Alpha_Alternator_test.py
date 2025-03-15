import pickle
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from utils import *  # Import utility functions
from AlphaAlt import *  # Import model-related functions

# Load training dataset
with open('dataset.p', 'rb') as file:
    dataset = pickle.load(file)

# Extract dataset components
xs = dataset['xs']  # Observations
vs = dataset['vendis']  # Velocity distributions
zs = dataset['latents']  # Latent variables

# Expand dimensions to ensure proper shape
xs = np.expand_dims(xs, axis=0)
zs = np.expand_dims(zs, axis=0)
vs = np.expand_dims(vs, axis=0)

# Set device for computations (CPU in this case)
device = torch.device('cpu')

# Prepare dataset and dataloader
Dataset_tr = get_dataset_HC(xs, zs, vs, device)
Dataset_loader_tr = DataLoader(Dataset_tr, batch_size=zs.shape[1], shuffle=False)

# Initialize model
model = AlphaAlt(latent_dim=1, obser_dim=xs.shape[-1], sigma_x=0.4, sigma_z=0.2, vendi_L=10, vendi_q=0.1,
                 importance_sample_size=1, n_layers=2, device=device).to(device)

# Print model parameter counts
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The Alpha-Alternator model has {count_parameters(model):,} trainable parameters')

# Set up optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
CLIP = 1  # Gradient clipping value
Numb_Epochs = 1000  # Number of training epochs

total_loss = []  # Store loss per epoch

# Training loop
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader_tr):
        x, z, v = batch  # Extract batch data
        
        optimizer.zero_grad()  # Reset gradients
        
        # Generate mask for missing values
        mask_imput = get_mask_imputation(x.shape[0], 30)
        
        # Sample noise for latent and observed variables
        eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
        eps_x = model.sigma_x * torch.randn(x[0].shape).to(device)
        
        # Forward pass
        z_hat = model(x, mask_imput, eps_x, eps_z, train=True)
        
        print(f'Epoch {epoch+1}/{Numb_Epochs}')
        
        # Compute loss
        loss = model.loss(a=1, b=1, c=1, z=z)
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # Apply gradient clipping
        optimizer.step()  # Update model parameters
        
        epoch_loss += loss.item()
    
    total_loss.append(epoch_loss)

# Save trained model
torch.save(model.state_dict(), 'Alpha_Alt_model.pt')

# Load model for evaluation
model.load_state_dict(torch.load('Alpha_Alt_model.pt', map_location=torch.device('cpu')))

# Load test dataset
with open('dataset_test.p', 'rb') as file:
    dataset = pickle.load(file)

# Extract test dataset components
xs_te = dataset['xs']
vs_te = dataset['vendis']
zs_te = dataset['latents']

# Expand dimensions for compatibility
xs_te = np.expand_dims(xs_te, axis=0)
vs_te = np.expand_dims(vs_te, axis=0)
zs_te = np.expand_dims(zs_te, axis=0)

# Prepare test dataset and dataloader
Dataset_te = get_dataset_HC(xs_te, zs_te, vs_te, device)
Dataset_loader_te = DataLoader(Dataset_te, batch_size=zs_te.shape[1], shuffle=False)

# Extract test batch
for i, batch in enumerate(Dataset_loader_te):
    x, z, v = batch

z = z.detach().cpu().numpy().squeeze()[1:]  # Convert z to NumPy format and remove first sample

# Number of trajectory samples to generate
trj_samples = np.arange(0, 10)
all_z_hats = []  # Store predicted latent values
all_x_hats = []  # Store predicted observation values

# Generate multiple trajectory samples
for ii in trj_samples:
    mask_imput = get_mask_forcasting(x.shape[0], 0)  # Generate forecasting mask
    
    eps_z = model.sigma_z * torch.randn(z[0].shape).to(device)
    eps_x = model.sigma_x * torch.randn(x[0].shape).to(device)
    
    z_hat = model(x, mask_imput, eps_x, eps_z, train=False)  # Model inference
    
    # Store predictions
    all_z_hats.append(z_hat.detach().cpu().numpy().squeeze()[1:])
    all_x_hats.append(z_hat.detach().cpu().numpy().squeeze()[1:])

# Convert predictions to NumPy arrays
all_z_hats = np.array(all_z_hats)
all_x_hats = np.array(all_x_hats)

# Plot results
plt.figure(figsize=(14, 4))
plt.plot(z, 'k', linewidth=2, label='True Latent')
plt.plot(all_z_hats.T, color='#D2A106', alpha=0.3, linewidth=1, label='Predicted Samples')
plt.plot(all_z_hats.T.mean(axis=-1), color='#D2A106', alpha=0.8, linewidth=2, label='Mean Prediction')
plt.title('Alpha Alt Model Predictions')
plt.legend()
plt.savefig('alpha_alt_simulation.pdf', format='pdf')
plt.show()

# Compute error metrics
MAE = np.abs(all_z_hats.T.mean(axis=-1) - z).mean()
MSE = ((all_z_hats.T.mean(axis=-1) - z) ** 2).mean()
pearson_correlation, _ = pearsonr(all_z_hats.T.mean(axis=-1), z)

print(f'MAE = {MAE}')
print(f'MSE = {MSE}')
print(f'Pearson Correlation (CC) = {pearson_correlation * 100:.2f}%')

# Save results
Alpha_Alt_results = {
    'all_z_hats': all_z_hats,
    'z': z,
    'pearson_correlation': pearson_correlation,
    'MAE': MAE,
    'MSE': MSE
}

pickle.dump(Alpha_Alt_results, open("Alpha_Alt_results.p", "wb"))
