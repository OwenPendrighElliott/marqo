import os

import open_clip
import torch
import numpy as np
from open_clip.pretrained import _pcfg, _slpcfg, _apcfg
from open_clip.transform import image_transform_v2, PreprocessCfg, merge_preprocess_dict
from pydantic.v1 import ValidationError
from torchvision.transforms import Compose
import ingrain
from marqo import marqo_docs
from marqo.core.inference.embedding_models.abstract_clip_model import AbstractCLIPModel
from marqo.core.inference.embedding_models.hf_tokenizer import HFTokenizer
from marqo.core.inference.embedding_models.open_clip_model_properties import OpenCLIPModelProperties, ImagePreprocessor
from marqo.core.inference.model_download import download_model
from marqo.exceptions import InternalError
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelAuth, ModelLocation

logger = get_logger(__name__)

HF_HUB_PREFIX = "hf-hub:"
MARQO_OPEN_CLIP_REGISTRY_PREFIX = "open_clip/"


class OPEN_CLIP(AbstractCLIPModel):
    def __init__(
            self,
            device: Optional[str] = None,
            model_properties: Optional[Dict] = None,
            model_auth: Optional[ModelAuth] = None,
    ) -> None:

        super().__init__(device=device, model_properties=model_properties, model_auth=model_auth)

        self.model_properties = self._build_model_properties(model_properties)
        self.preprocess_config = None

        client = ingrain.Client()
        self.model_name = self.model_properties.name
        self.pretrained = self.model_properties.pretrained

        client.load_clip_model(name=self.model_name, pretrained=self.pretrained)

    def _build_model_properties(self, model_properties: dict) -> OpenCLIPModelProperties:
        """Convert the user input model_properties to OpenCLIPModelProperties."""
        try:
            return OpenCLIPModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties: {model_properties}. Original error: {e}") \
                from e

    def _load_necessary_components(self) -> None:
        pass
    def _check_loaded_components(self):
        pass

    def _load_image_preprocessor(self) -> Callable:
        return lambda x: x
    def _aggregate_image_preprocessor_config(self) -> PreprocessCfg:
        return PreprocessCfg()

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]],
                     media_download_headers: Optional[Dict] = None,
                     normalize=True) -> FloatTensor:
        client = ingrain.Client(return_numpy=True, user_agent=media_download_headers.get('User-Agent', 'Marqobot/1.0'))
        response = client.infer_image(name=self.model_name, pretrained=self.pretrained, image=images, normalize=normalize)
        return np.array(response["embeddings"])
        

    def encode_text(self, sentence: Union[str, List[str]], normalize=True) -> FloatTensor:
        client = ingrain.Client(return_numpy=True)
        response = client.infer_text(name=self.model_name, pretrained=self.pretrained, text=sentence, normalize=normalize)

        return np.array(response["embeddings"])