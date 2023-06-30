# https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu

import io
import pickle
import torch

class CPU_Unpickler(pickle.Unpickler):
    '''
    Helper class for loading PyTorch models (GPU Trained, CPU Load)
    '''
    def find_class(self, module, name):
        '''
        Load .pkl model, cpu/gpu option
        Args:
            None
        Returns:
            None
        '''
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
