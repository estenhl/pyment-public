from .model import Model
from .model_type import ModelType
from .ranking_sfcn import RankingSFCN
from .regression_sfcn import RegressionSFCN
from .soft_classification_sfcn import SoftClassificationSFCN
from .vit import RegressionVisionTransformerMini, VisionTransformer


def get(model_name: str, **kwargs) -> Model:
    if model_name.lower() in ['regressionsfcn', 'sfcn-reg']:
        return RegressionSFCN(**kwargs)
    elif model_name.lower() in ['softclassificationsfcn', 'sfcn-sm']:
        return SoftClassificationSFCN(**kwargs)
    elif model_name.lower() in ['rankingsfcn', 'sfcn-rank']:
        return RankingSFCN(**kwargs)
    elif model_name.lower() in ['regressionvisiontransformermini',
                                'vis-mini-reg']:
        return RegressionVisionTransformerMini(**kwargs)
    else:
        raise ValueError(f'Unknown model {model_name}')

def get_model_names() -> str:
    return ['sfcn-reg', 'sfcn-sm', 'sfcn-rank', 'vis-mini-reg']
