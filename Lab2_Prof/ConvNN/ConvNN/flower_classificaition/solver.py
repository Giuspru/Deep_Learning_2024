import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from ConvNN.flower_classificaition.net import Net
from ConvNN.flower_classificaition.dataset import Dataset

class Solver():
    def __init__(self, args):
        # prepare a dataset
        self.train_data = Dataset(train=True,
                                  data_root=args.data_root,
                                  size=args.image_size)
        self.test_data  = Dataset(train=False,
                                  data_root=args.data_root,
                                  size=args.image_size)
        #TODO create data loader
        
        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #TODO instatiate model, loss function and optimization strategy
        #TODO self.net=....
        # TODO loss=....
        # TODO loss=....
        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            #TODO implement training loop for each epooch
            
            if (epoch+1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_data)
                test_acc  = self.evaluate(self.test_data)

                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                    format(epoch+1, args.max_epochs, loss.item(), train_acc, test_acc))
                
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False)

        self.net.eval()
        num_correct, num_total = 0, 0
        
        with torch.no_grad():
            for inputs in loader:
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)

                outputs = self.net(images)
                _, preds = torch.max(outputs.detach(), 1)

                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
