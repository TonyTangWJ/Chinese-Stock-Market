import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1, use_sigmoid = False,model_name = "linear_regression"):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.use_sigmoid = use_sigmoid
        self.model_name = model_name
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def reset_model(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        return

    def forward(self, x):
        x = self.linear(x)
        if self.use_sigmoid:
            x = nn.Sigmoid()(x)
        return x
    
    def train_step(self, train_loader, criterion=None):
        if criterion is None:
            criterion = self.loss_fn
        self.train()
        total_loss = 0.0
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[3].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=100, resume_training=False, patience=5):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self.load_checkpoint()
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
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping triggered due to no improvement in loss.")
                    break
        self.save_model()

    # Predict method & reverse normalization
    def predict(self, data, label_mean_val, label_std_val):
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = self.forward(data)
        # reverse normalization
        pred = pred * label_std_val + label_mean_val
        return pred
        
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[3].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
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
        print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        print (f"Model successfully loaded from {path}")
