from rbf_polytope import RbfModel
import numpy as np

def save_model(model:RbfModel, path:str):
    np.savez_compressed(path,
                        centers=model.centers,
                        widths=model.widths,
                        vertices=model.vertices,
                        center_norm=model.center_norm,
                        is_continuous=model.is_continuous)

def load_model(path:str)->RbfModel:
    data = np.load(path)
    return RbfModel(centers=data['centers'],
                    widths=data['widths'],
                    vertices=data['vertices'],
                    center_norm=data['center_norm'],
                    is_continuous=data['is_continuous'])
