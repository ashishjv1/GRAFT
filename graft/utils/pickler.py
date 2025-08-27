import pickle
import os

class pickler():

    def save_pickle(V, dirs, bs) -> None:
        folder_name = f"{dirs}" + "_pickle"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        
        file = os.path.join(f"{dirs}" + "_pickle", f"V_{bs}.pkl") 
        with open(file, 'wb') as f:
             pickle.dump(V, f)
                
        if os.path.exists(file):
            print("pickle saved successfully")
        else:
            print("Failed to save pickle.")


    def load_pickle(dirs, bs):
                
        file = os.path.join(f"{dirs}" + "_pickle", f"V_{bs}.pkl") 

        with open(file, 'rb') as f:
            data3 = pickle.load(f)
        return data3