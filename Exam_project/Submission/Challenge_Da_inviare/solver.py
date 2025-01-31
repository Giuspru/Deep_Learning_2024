import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DepthDataset
from utils import visualize_img, ssim
from model import Net


class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args

        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN, #instanziamento dataset di tain
                                           data_dir=args.data_dir,
                                           transform=None)
            
            self.val_data = DepthDataset(train=DepthDataset.VAL, #instanziamento dataset di Validation
                                         data_dir=args.data_dir,
                                         transform=None)

            self.train_dataloader = DataLoader(dataset= self.train_data,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               drop_last=True)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.loss_mse = torch.nn.MSELoss()
            self.alpha = 0.65
            self.beta = 0.35

            self.net = Net().to(self.device)
            self.optim = torch.optim.Adam(self.net.parameters(), self.args.lr)

            '''Questo if-else qua sopra distingue tra caso di training e caso di test 
                - A quanto pare sembrerebbe che io debba implementare tutto il mio codice nel primo blocco e non nell'else.
                - Instanzia da solo train e validation
                - La funzione sotto non dovrebbe mai farla perchè in teoria ho già la cartella checkpoint
               Il blocco Else servirà al professore per quando testerà il suo codice, pertantod non dovremmo toccarlo. Se però volessi vedere se il mio codice 
               funziona adeguatamentequando lui lo eseguirà in test mi basta modificare DepthDataset.TEST in DepthDataset.VAL. In pratica faccio test dul validation set, ma mi serve solo per vedere 
               se il codice funzionerà al professore.
                   -In questo caso nel main devi specificare is_train = False

            '''

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST,  #<----------------------------------------------------------------- qui ci puoi mettere DepthDataset.VAL invece di DepthDataset.TEST
                                    data_dir=self.args.data_dir)
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = Net().to(self.device)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True)) #<-- qui c'è il load del modello: faccio load state dict dal path del checkpoint, quindi carico un checkpoint salvato su disco e mi carico i parametri sulla rete


    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):

            self.net.train()

            for step, inputs in enumerate(self.train_dataloader):
                rgb = inputs[0].to(self.device)
                depth = inputs[1].to(self.device)

                pred = self.net(rgb)

                ssim_value = 1 - ssim(pred, depth)
                mse_value = self.loss_mse(pred, depth)
                loss = self.alpha * ssim_value + self.beta * mse_value

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            print("Epoch [{}/{}] Loss: {:.3f} ".format(epoch + 1, args.max_epochs, loss.item()))
            self.save(ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name, global_step=epoch)
            self.evaluate(DepthDataset.VAL)

        


    '''
        Un'altra funzione che trovo implementata p evaluate(), che mi spiega come viene effettuata la evaluation della rete, di fatto questa la posso usare durante il training 
        per fare il test del modello sul training e sul validation per vedere come sta andado il mio addestramento.
        Serve per il model assessmment.

        Evaluate distingue tra train e validation, praticamente crea un dataloader usando uno dei due dataset, e poi si calcola le metriche che valutano quanto le depth map stimate
        sono simili aquelle di ground truth.

        Possiamo chiamare evaluate alla fine di pgni epoca per vedere come sta andando il training. siamo liberi di usarla la funzione
    '''
    def evaluate(self, set):

        args = self.args
        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))

    '''
       Viene implementata anche la save() del checkpoint, e questa la posso chiamare ad ogni epoca per salvare il modello corrente,
       Il load invece lo troviamo nell'else per il training. riga 43
    '''
    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

    '''
    Questa la userà il professore perchè è diversa da evaluate, e la chiamo quando fa test del mio modello.
    Potrei testare la chiamata di questa funzione semplicemente modificando al posto di self.test_set self.val_set
    Ricordati che poi lo devo ricambiare primoa di rimandare il progetto.
    '''
    def test(self):

        loader = DataLoader(self.test_set, 
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
