import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
from utils import ClippedLeakyReLU, ScaledTanh

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1, target =3, model_name = "linear_regression"):
        super(LinearRegression, self).__init__()
        
        self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim)
            )
        
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.target = target
        self.model_name = model_name
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)
    

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1)
                nn.init.constant_(module.bias, 0)

    
    def train_step(self, train_loader,criterion=None):
        if criterion is None:
            criterion = nn.MSELoss()
        elif criterion == "elastic_net":
            criterion = self.elastic_net_loss
        else:
            raise ValueError("Unsupported criterion. Supported values are 'elastic_net', 'mse_rs', or None (default MSE).")

        self.train()
        total_loss = 0.0
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[self.target].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


    def fit(self, train_loader, epochs=100, resume_training=False, patience=5, criterion=None):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # if loss is not improving for 5 epochs, stop training
            if round(train_loss,4) < round(prev_loss,4):
                prev_loss = train_loss
                no_improvement_count = 0  
                self.save_model()  
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered due to no improvement in loss.")
                    break
        

    # Predict method & reverse normalization
    def predict(self, test_loader, label_mean, label_std):
        self.eval()
        pred_list = []
        act_list = []
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                if data.dim() == 1:
                    data = data.unsqueeze(1)
                pred = self(data)
                # reverse normalization
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act


    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[self.target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # targets >=0
                targets = torch.clamp(targets, min=0)
                outputs = torch.clamp(outputs, min=0)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

  
    def elastic_net_loss(self, outputs, targets):
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = 0
        l2_reg = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_reg += torch.sum(torch.abs(module.weight))
                l2_reg += torch.sum(module.weight ** 2)
        
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return mse_loss + elastic_reg


    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)
        # print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        # print (f"Model successfully loaded from {path}")




class ElasticNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, target=3, alpha=1.0, l1_ratio=0.5, model_name = "ElasticNet"):
        super(ElasticNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.model_name = model_name
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.target = target
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        return self.linear(x)

    def elastic_net_loss(self, outputs, targets):
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = torch.sum(torch.abs(self.linear.weight))
        l2_reg = torch.sum(self.linear.weight ** 2)
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return mse_loss + elastic_reg

    def reset_model(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        return

    def train_step(self, train_loader,criterion=None):
        if criterion is None:
            criterion = self.elastic_net_loss
        self.train()
        total_loss = 0.0
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[self.target].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=100, resume_training=False, patience=5, criterion=None):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # if loss is not improving for 5 epochs, stop training
            if round(train_loss,4) < round(prev_loss,4):
                prev_loss = train_loss
                no_improvement_count = 0   
                self.save_model() 
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered due to no improvement in loss.")
                    break
        

    # Predict method & reverse normalization
    def predict(self, test_loader, label_mean, label_std):
        self.eval()
        pred_list = []
        act_list = []
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                if data.dim() == 1:
                    data = data.unsqueeze(1)
                pred = self(data)
                # reverse normalization
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act
        
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[self.target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # targets >=0
                targets = torch.clamp(targets, min=0)
                outputs = torch.clamp(outputs, min=0)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)
        # print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        # print (f"Model successfully loaded from {path}")
    

class NN(nn.Module):
    def __init__(self, input_dim, output_dim=1, target=3, alpha=1.0, l1_ratio=0.5, layer = 2, model_name = "NN"):
        super(NN, self).__init__()
        
        if layer == 1:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim)
            )
        elif layer == 2:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, output_dim)
            )
        elif layer == 3:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, output_dim)
            )
        elif layer == 4:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, output_dim)
            )
        elif layer == 5:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(0.1),
                nn.Linear(4, output_dim)
            )
        else:
            raise ValueError("Unsupported number of layers. Supported values are 1 to 5.")
        
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.model_name = model_name + f"_{layer}Layers"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.target = target
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.layers(x)

    def elastic_net_loss(self, outputs, targets):
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = 0
        l2_reg = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_reg += torch.sum(torch.abs(module.weight))
                l2_reg += torch.sum(module.weight ** 2)
        
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return mse_loss + elastic_reg


    def reset_model(self):
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        return


    def train_step(self, train_loader, criterion=None):
        if criterion is None:
            criterion = self.loss_fn
        elif criterion == "elastic_net":
            criterion = self.elastic_net_loss
        else:
            raise ValueError("Unsupported criterion. Supported values are 'elastic_net', 'mse_rs', or None (default MSE).")

        self.train()
        total_loss = 0.0

        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[self.target].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


    def fit(self, train_loader, epochs=100, resume_training=False, patience=5, criterion=None):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader, criterion)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # if loss is not improving for 5 epochs, stop training
            if round(train_loss,4) < round(prev_loss,4):
                prev_loss = train_loss
                no_improvement_count = 0  
                # 保存最佳模型
                self.save_model()  
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered due to no improvement in loss.")
                    break
        
    # Predict method & reverse normalization
    def predict(self, test_loader, label_mean, label_std):
        self.eval()
        pred_list = []
        act_list = []
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                if data.dim() == 1:
                    data = data.unsqueeze(1)
                pred = self(data)
                # reverse normalization
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act
        
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[self.target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # targets >=0
                targets = torch.clamp(targets, min=0)
                outputs = torch.clamp(outputs, min=0)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

    # def evaluate(self, test_loader):
    #     self.eval()
    #     total_sse = 0.0
    #     total_ss = 0.0
    #     all_outputs = []
    #     all_targets = []
        
    #     with torch.no_grad():
    #         for data in test_loader:
    #             inputs = data[0].to(self.device)
    #             targets = data[self.target].to(self.device)
                
    #             if targets.dim() == 1:
    #                 targets = targets.unsqueeze(1)
    #             outputs = self(inputs)

    #             targets = torch.clamp(targets, min=0)
    #             outputs = torch.clamp(outputs, min=0)
    #             # 计算指标
    #             sse = torch.sum((targets - outputs) ** 2)
    #             ss = torch.sum(targets ** 2)
    #             total_sse += sse.item()
    #             total_ss += ss.item()
                
    #             # 保存结果
    #             all_outputs.append(outputs.cpu())
    #             all_targets.append(targets.cpu())
        
    #     # 计算R²
    #     r_squared = 0 if total_ss < 1e-8 else 1 - total_sse / total_ss
        
    #     # 保存预测结果
    #     if all_outputs and all_targets:
    #         self._save_predictions(all_outputs, all_targets)
        
    #     return r_squared
    
    # def _save_predictions(self, outputs, targets):
    #     combined_outputs = torch.cat(outputs, dim=0).numpy().flatten()
    #     combined_targets = torch.cat(targets, dim=0).numpy().flatten()
        
    #     os.makedirs('../CSV', exist_ok=True)
    #     np.savetxt('../CSV/outputs_targets.csv', 
    #               np.column_stack((combined_outputs, combined_targets)),
    #               delimiter=',', 
    #               header='outputs,targets', 
    #               comments='',
    #               fmt='%.4f')


    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)
        # print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        # print (f"Model successfully loaded from {path}")


class RandomForest:
    def __init__(self, target=3, n_estimators=100, max_depth=None, min_samples_split=5, 
                 min_samples_leaf=2, random_state=42, model_name="RandomForest"):
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,      # 树的数量
            max_depth=max_depth,            # 树的最大深度
            min_samples_split=min_samples_split,  # 分割节点最小样本数
            min_samples_leaf=min_samples_leaf,    # 叶节点最小样本数
            random_state=random_state,      # 随机种子
            n_jobs=-1                      # 使用所有CPU核心
        )
        
        self.model_name = model_name
        self.feature_names = None
        self.target = target
        
        # 创建模型保存目录
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
            
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pkl"
        self.model_path = f"model/final_models/{self.model_name}.pkl"

    def fit(self, train_loader, epochs=None, patience=None, criterion=None):
        """
        训练随机森林模型
        注意：随机森林不需要epochs和patience参数
        """
        X_train, y_train = self._extract_data_from_loader(train_loader)
        self.model.fit(X_train, y_train)
        self.save_model()
        print(f"Random Forest trained with {len(X_train)} samples")

    def predict(self, test_loader, label_mean=None, label_std=None):
        """预测并返回与PyTorch模型相同的格式"""
        X_test, y_test = self._extract_data_from_loader(test_loader)
        pred = self.model.predict(X_test)
        
        # 转换为与PyTorch模型相同的格式
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
            
        return pred, y_test

    def evaluate(self, test_loader):
        """计算 nondemeaned R²分数"""
        X_test, y_test = self._extract_data_from_loader(test_loader)
        pred = self.model.predict(X_test)
        
        # 确保预测值和目标值非负（与其他模型保持一致）
        y_test = np.clip(y_test, a_min=0, a_max=None)
        pred = np.clip(pred, a_min=0, a_max=None)
        
        # 计算 nondemeaned R²
        # R² = 1 - SSE/SS (其中 SS = Σ(yi)², 不减去均值)
        sse = np.sum((y_test - pred) ** 2)  # 残差平方和
        ss = np.sum(y_test ** 2)            # 总平方和（不去均值）
        
        if ss < 1e-8:
            return 0
        else:
            return 1 - sse / ss

    def _extract_data_from_loader(self, data_loader):
        """从PyTorch DataLoader中提取数据"""
        X_list = []
        y_list = []
        
        for batch in data_loader:
            X_batch = batch[0].numpy()  # 特征
            y_batch = batch[self.target].numpy()  # 目标
            
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # 确保y是一维数组（sklearn期望的格式）
        if y.ndim > 1:
            y = y.ravel()
            
        return X, y

    def reset_model(self):
        """重置模型（重新初始化）"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 保存原始参数
        params = self.model.get_params()
        self.model = RandomForestRegressor(**params)
        print("Random Forest model reset")

    def save_model(self, path=None):
        """保存模型"""
        import joblib
        if path is None:
            path = self.model_path
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model successfully saved to {path}")

    def load_model(self, path=None):
        """加载模型"""
        import joblib
        if path is None:
            path = self.model_path
            
        if os.path.exists(path):
            self.model = joblib.load(path)
            # print(f"Model successfully loaded from {path}")
        else:
            print(f"Model file {path} not found")

    def get_feature_importance(self):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print("Model not trained yet")
            return None
    

class K_Means_NN(nn.Module):
    def __init__(self, input_dim, output_dim=1, target=3, n_clusters=10, layer=2, alpha=1.0, l1_ratio=0.5, model_name="K_Means_NN"):
        super(K_Means_NN, self).__init__()
        
        from sklearn.cluster import KMeans
        
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.model_name = model_name + f"_{layer}Layers_{n_clusters}Clusters"
        
        # K-means聚类器
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,      # 聚类数量
            init='k-means++',           # 初始化方法（'k-means++', 'random'）
            n_init='auto',                  # 不同初始化的运行次数
            max_iter=300,               # 单次运行的最大迭代次数
            tol=1e-4,                   # 收敛容忍度
            random_state=42,            # 随机种子
            algorithm='lloyd',          # 算法类型（'lloyd', 'elkan'）
            copy_x=True                # 是否复制数据
        )
        self.cluster_labels = None
        self.is_fitted = False
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.target = target
        
        # 为每个聚类创建独立的神经网络
        self.cluster_models = nn.ModuleDict()
        for i in range(n_clusters):
            self.cluster_models[f'cluster_{i}'] = self._create_network(input_dim, output_dim, layer)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        
        # 设备设置
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        
        # 模型保存路径
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"

    def _create_network(self, input_dim, output_dim, layer):
        """为每个聚类创建独立的神经网络"""
        if layer == 1:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim)
            )
        elif layer == 2:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, output_dim)
            )
        elif layer == 3:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, output_dim)
            )
        elif layer == 4:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, output_dim)
            )
        elif layer == 5:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(0.1),
                nn.Linear(4, output_dim)
            )
        else:
            raise ValueError("Unsupported number of layers. Supported values are 1 to 5.")
        
        self._initialize_weights()
    
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def fit_kmeans(self, train_loader):
        """训练K-means聚类器"""
        print("Fitting K-means clustering...")
        
        # 提取所有训练数据
        X_all = []
        for batch in train_loader:
            X_batch = batch[0].cpu().numpy()
            X_all.append(X_batch)
        
        X_all = np.concatenate(X_all, axis=0)
        
        # 训练K-means
        self.cluster_labels = self.kmeans.fit_predict(X_all)
        self.is_fitted = True
        
        # 打印聚类信息
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"Cluster {cluster_id}: {count} samples ({count/len(X_all)*100:.1f}%)")
        
        return self.cluster_labels

    def create_cluster_dataloaders(self, original_dataloader):
        """为每个聚类创建独立的DataLoader，保持原始batch_size"""
        if not self.is_fitted:
            self.fit_kmeans(original_dataloader)
        
        from torch.utils.data import TensorDataset, DataLoader
        
        # 提取所有数据
        all_data = []
        for batch in original_dataloader:
            all_data.append(batch)
        
        # 按聚类分组数据
        cluster_dataloaders = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_samples = []
            sample_idx = 0
            
            for batch in all_data:
                batch_size = batch[0].size(0)
                
                for i in range(batch_size):
                    if sample_idx + i < len(self.cluster_labels):
                        if self.cluster_labels[sample_idx + i] == cluster_id:
                            # 收集该样本
                            sample = [batch[j][i] for j in range(len(batch))]
                            cluster_samples.append(sample)
                
                sample_idx += batch_size
            
            if cluster_samples:
                # 创建该聚类的TensorDataset
                cluster_tensors = []
                for j in range(len(cluster_samples[0])):
                    tensor_data = torch.stack([sample[j] for sample in cluster_samples])
                    cluster_tensors.append(tensor_data)
                
                cluster_dataset = TensorDataset(*cluster_tensors)
                cluster_dataloader = DataLoader(
                    cluster_dataset, 
                    batch_size=original_dataloader.batch_size
                )
                cluster_dataloaders[cluster_id] = cluster_dataloader
    
        return cluster_dataloaders

    def train_step(self, train_loader, criterion=None):
        """训练步骤 - 使用固定batch size的聚类数据"""
        if not self.is_fitted:
            self.fit_kmeans(train_loader)
        
        if criterion is None:
            criterion = self.loss_fn
        elif criterion == 'elastic_net':
            criterion = self.elastic_net_loss
        else:
            raise ValueError("Unsupported criterion. Supported values are 'elastic_net', 'mse_rs', or None (default MSE).")
        
        self.train()
        total_loss = 0.0
        total_samples = 0
        
        # 创建每个聚类的DataLoader（只在第一次调用时创建）
        if not hasattr(self, 'cluster_dataloaders') or not self.cluster_dataloaders:
            print("Creating cluster dataloaders with fixed batch size...")
            self.cluster_dataloaders = self.create_cluster_dataloaders(train_loader)
        
        # 为每个聚类分别训练
        for cluster_id in range(self.n_clusters):
            if cluster_id not in self.cluster_dataloaders:
                print(f"Cluster {cluster_id}: No data or insufficient samples")
                continue
                
            cluster_dataloader = self.cluster_dataloaders[cluster_id]
            cluster_loss = 0.0
            cluster_samples = 0
            
            for batch in cluster_dataloader:
                inputs = batch[0].to(self.device)
                targets = batch[self.target].to(self.device)
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                self.optimizer.zero_grad()
                
                # 使用对应聚类的模型
                outputs = self.cluster_models[f'cluster_{cluster_id}'](inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                cluster_loss += loss.item()
                cluster_samples += len(inputs)
            
            if cluster_samples > 0:
                total_loss += cluster_loss
                total_samples += cluster_samples
                # print(f"Cluster {cluster_id}: Loss = {cluster_loss/len(cluster_dataloader):.4f}, Samples = {cluster_samples}")
        
        return total_loss / max(total_samples, 1)

    def forward(self, x, cluster_id=None):
        """前向传播"""
        if cluster_id is not None:
            return self.cluster_models[f'cluster_{cluster_id}'](x)
        else:
            # 如果没有指定聚类，需要先预测聚类
            x_np = x.cpu().numpy()
            predicted_clusters = self.kmeans.predict(x_np)
            
            outputs = []
            for i, cluster in enumerate(predicted_clusters):
                sample_input = x[i:i+1]
                output = self.cluster_models[f'cluster_{cluster}'](sample_input)
                outputs.append(output)
            
            return torch.cat(outputs, dim=0)

    def predict(self, test_loader, label_mean, label_std):
        """预测方法"""
        self.eval()
        pred_list = []
        act_list = []
        
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                
                # 提取实际值
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                
                # 预测
                pred = self.forward(data)
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act

    def fit(self, train_loader, epochs=50, resume_training=False, patience=5, criterion=None):
        """训练主函数 - 添加checkpoint支持"""
        
        # 如果需要恢复训练，先加载checkpoint
        start_epoch = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")
        else:
            # 首先训练K-means（只在新训练时）
            if not self.is_fitted:
                self.fit_kmeans(train_loader)
            # self._initialize_weights()
        
        no_improvement_count = 0
        prev_loss = float('inf')
        
        # print(f"Training {self.model_name} with {self.n_clusters} clusters...")
        
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader, criterion=criterion)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")
            # 每10个epoch保存checkpoint和打印信息
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")
            
            # 早停机制
            if round(train_loss, 4) < round(prev_loss, 4):
                prev_loss = train_loss
                no_improvement_count = 0
                # 保存最佳模型
                self.save_model()
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Training completed for {self.model_name}")
        self.is_fitted = False

    def save_model(self, path=None):
        """保存最终模型"""
        if path is None:
            path = self.model_path
        
        save_dict = {
            'state_dict': self.state_dict(),
            'kmeans': self.kmeans,
            'cluster_labels': self.cluster_labels,
            'is_fitted': self.is_fitted,
            'n_clusters': self.n_clusters,
            'model_name': self.model_name,
            'input_dim': self.input_dim
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        # print(f"Model successfully saved to {path}")

    def load_model(self, path=None):
        """加载最终模型"""
        if path is None:
            path = self.model_path
        
        if os.path.exists(path):
            try:
                save_dict = torch.load(path, map_location=self.device, weights_only=False)
                
                self.load_state_dict(save_dict['state_dict'])
                self.kmeans = save_dict['kmeans']
                self.cluster_labels = save_dict['cluster_labels']
                self.is_fitted = save_dict['is_fitted']
                
                # print(f"Model successfully loaded from {path}")
                return True
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                return False
        else:
            print(f"Model file {path} not found")
            return False

    def reset_model(self):
        """重置模型"""
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.is_fitted = False
        self.cluster_labels = None
        # 清除缓存的cluster dataloaders
        if hasattr(self, 'cluster_dataloaders'):
            delattr(self, 'cluster_dataloaders')

    def get_cluster_info(self):
        """获取聚类信息"""
        if not self.is_fitted:
            return "K-means not fitted yet"
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        info = f"K-means clustering with {self.n_clusters} clusters:\n"
        for cluster_id, count in zip(unique, counts):
            info += f"  Cluster {cluster_id}: {count} samples ({count/len(self.cluster_labels)*100:.1f}%)\n"
        return info
    
    def _save_checkpoint(self, epoch):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
            'kmeans': self.kmeans,
            'cluster_labels': self.cluster_labels,
            'is_fitted': self.is_fitted,
            'n_clusters': self.n_clusters,
            'model_name': self.model_name
        }
        
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        """加载训练检查点"""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                
                # 加载模型状态
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 加载K-means相关信息
                self.kmeans = checkpoint['kmeans']
                self.cluster_labels = checkpoint['cluster_labels']
                self.is_fitted = checkpoint['is_fitted']
                
                epoch = checkpoint['epoch']
                print(f"Checkpoint loaded successfully from epoch {epoch + 1}")
                return epoch
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch.")
                return 0
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0
    
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[self.target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # targets >=0
                targets = torch.clamp(targets, min=0)
                outputs = torch.clamp(outputs, min=0)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss
        
    def elastic_net_loss(self, outputs, targets):
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = 0
        l2_reg = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_reg += torch.sum(torch.abs(module.weight))
                l2_reg += torch.sum(module.weight ** 2)
        
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return mse_loss + elastic_reg
