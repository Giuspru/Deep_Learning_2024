import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001) #modificabile
    parser.add_argument("--batch_size", type=int, default=50) #modificabile
    parser.add_argument("--max_epochs", type=int, default=25) #modificabile
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint") #non modiifcare
    parser.add_argument("--ckpt_name", type=str, default="depth") #non modificare
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--visualize_every", type=int, default=100) 
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(".", "DepthEstimationUnreal")) #cambiare la parte per il percorso al mio stesso dataset tranne l'ultimo pezzo

    parser.add_argument("--is_train", type=bool, default=True)  #<------------------------------------------------Verifica se funziona al professore
    parser.add_argument("--ckpt_file", type=str, default="depth_60.pth")

    args = parser.parse_args()
    solver = Solver(args)
    if args.is_train:
        solver.fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()


'''
per quanto riguarda il parametro --evaluate_every posso modificarlo, per vedere ogni quanto fare l'evaluation, se voglio usare questo parametro nel loop di addestramento
serve per dire fai il test sul training e sul validation ogni tot epoche.

lines 18 and 19: 
cambia la prima riga per testare il tuo modello, così praticamente accedi al test dell'if-else che avevamo prima
per farlo devi: cambiare la variabile --is_train in False e cambiare la variabile --ckpt_file in un checkpoint che hai salvato prima

/Users/giuspru/Desktop/Challengegiuseppe/checkpoint <-- path dove salvo i checkpoint
/Users/giuspru/Desktop/Challengegiuseppe/DepthEstimationUnreal <-- path del dataset

'''