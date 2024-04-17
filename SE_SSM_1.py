import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

class AttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_cities):
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_cities = num_cities
        
        # 时间维度注意力
        self.temporal_query = nn.Linear(input_dim, hidden_dim * num_heads)
        self.temporal_key = nn.Linear(input_dim, hidden_dim * num_heads)
        self.temporal_value = nn.Linear(input_dim, hidden_dim * num_heads)
        
        # 城市间注意力(类似SE模块)
        self.fc_squeeze = nn.Linear(hidden_dim * num_heads, (hidden_dim * num_heads) // 2)
        self.fc_excitation = nn.Linear((hidden_dim * num_heads) // 2, hidden_dim * num_heads)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(hidden_dim * num_heads, input_dim)

    def forward(self, x):
        batch_size, seq_len, num_cities = x.size()
    
        # 时间维度注意力
        temporal_queries = self.temporal_query(x).view(batch_size, seq_len, num_cities, self.num_heads, self.hidden_dim).transpose(2, 3)
        temporal_keys = self.temporal_key(x).view(batch_size, seq_len, num_cities, self.num_heads, self.hidden_dim).transpose(2, 3)
        temporal_values = self.temporal_value(x).view(batch_size, seq_len, num_cities, self.num_heads, self.hidden_dim).transpose(2, 3)
        temporal_energy = torch.einsum("bqchd,bkchd->bhqck", [temporal_queries, temporal_keys])
        temporal_attention = torch.softmax(temporal_energy / (self.hidden_dim ** 0.5), dim=-1)
        temporal_out = torch.einsum("bhqck,bkchd->bqchd", [temporal_attention, temporal_values]).transpose(2, 3).reshape(batch_size, seq_len, num_cities, -1)
        
        # 城市间注意力(类似SE模块)
        spatial_squeeze = torch.mean(temporal_out, dim=1)
        spatial_squeeze = spatial_squeeze.view(batch_size, -1)
        spatial_squeeze = self.fc_squeeze(spatial_squeeze)
        spatial_excitation = self.sigmoid(self.fc_excitation(spatial_squeeze))
        spatial_excitation = spatial_excitation.view(batch_size, 1, num_cities, -1)
        spatial_out = temporal_out * spatial_excitation
        
        out = self.fc(spatial_out)
        return out

class StateSpaceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, variational=False):
        super(StateSpaceModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.variational = variational
        
        observation_layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            observation_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        observation_layers.append(nn.Linear(hidden_dim, output_dim))
        self.observation = nn.Sequential(*observation_layers)
        
        self.transition = nn.Linear(hidden_dim, hidden_dim)
        
        if self.variational:
            self.mean = nn.Parameter(torch.zeros(hidden_dim))
            self.log_var = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, state):
        observation = self.observation(x)
        if self.variational:
            state = self.transition(state) + torch.randn_like(state) * torch.exp(self.log_var) + self.mean
        else:
            state = self.transition(state)
        return observation, state


class OzoneModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_ssm_modules, num_ssm_layers, num_cities, variational=False):
        super(OzoneModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_ssm_modules = num_ssm_modules
        self.num_cities = num_cities
        self.attention = AttentionModule(input_dim, hidden_dim, num_heads, num_cities)
        self.ssm_modules = nn.ModuleList([StateSpaceModel(input_dim, hidden_dim, output_dim, num_ssm_layers, variational=variational) for _ in range(num_ssm_modules)])

    def forward(self, x, states):
        attended_x = self.attention(x)
        observations = []
        new_states = []
        for i, ssm_module in enumerate(self.ssm_modules):
            state_i = states[i].view(x.size(0), -1)
            state_i = state_i.unsqueeze(1).repeat(1, self.num_cities, 1)
            observation, new_state = ssm_module(attended_x, state_i)
            observations.append(observation)
            new_states.append(new_state.view(x.size(0), -1))
        observation = torch.stack(observations).mean(dim=0)
        return (observation,) + tuple(new_states)

data = np.load('target.npy') 
num_cities = data.shape[1]  
num_days = data.shape[0]

# 设置模型超参数
input_dim = 1
hidden_dim = 64
output_dim = 1
num_heads = 8
num_ssm_modules = 4
num_ssm_layers = 5
seq_len = 30
batch_size = 32

# 将数据重塑为 (num_days, num_cities, 1)
data = data.reshape(num_days, num_cities, 1)

# 将数据分为训练集、验证集和测试集（按时间划分）
train_ratio = 0.8
val_ratio = 0.1
train_size = int(num_days * train_ratio)
val_size = int(num_days * val_ratio)
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

# 转换为PyTorch张量
train_data = torch.from_numpy(train_data).float()
val_data = torch.from_numpy(val_data).float()
test_data = torch.from_numpy(test_data).float()

# 使用DataLoader加载数据
train_dataset = TensorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OzoneModel(input_dim, hidden_dim, output_dim, num_heads, num_ssm_modules, num_ssm_layers, num_cities, variational=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

# 训练模型
num_epochs = 800

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_data in train_loader:
        batch_data = batch_data[0].to(device)
        states = [torch.zeros(batch_data.size(0), hidden_dim).to(device) for _ in range(num_ssm_modules)]
        optimizer.zero_grad()
        
        preds = model(batch_data, states)[0]
        loss = criterion(preds.view(batch_data.size()), batch_data)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data[0].to(device)
            states = [torch.zeros(batch_data.size(0), hidden_dim).to(device) for _ in range(num_ssm_modules)]
            preds = model(batch_data, states)[0]
            loss = criterion(preds.view(batch_data.size()), batch_data)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 测试模型
model.eval()
test_data = test_data.to(device)
state1 = torch.zeros(test_data.size(0), hidden_dim).to(device)
state2 = torch.zeros(test_data.size(0), hidden_dim).to(device)
state3 = torch.zeros(test_data.size(0), hidden_dim).to(device)
state4 = torch.zeros(test_data.size(0), hidden_dim).to(device)
with torch.no_grad():
    test_preds, _, _, _, _ = model(test_data, [state1, state2, state3, state4])

# 计算各种评价指标
y_true = test_data.cpu().numpy().flatten()
y_pred = test_preds.cpu().numpy().flatten()

rmse = math.sqrt(mean_squared_error(y_true, y_pred))
print(f'RMSE: {rmse:.4f}')

mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae:.4f}')

mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  
print(f'MAPE: {mape:.2f}%')

r2 = r2_score(y_true, y_pred)
print(f'R^2 score: {r2:.4f}')

# 绘制预测值和真实值的对比图
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Ozone Level')
plt.title('Ozone Level Prediction')
plt.show()
