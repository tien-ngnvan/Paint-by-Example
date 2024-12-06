import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

import requests
import torch
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer
import torch.nn.functional as f


class FrozenJinaCLIPImageEmbedder():

    def __init__(
        self,
        model_name_or_path: str = 'jinaai/jina-clip-v2',
        device: str = 'cuda:0',
        **_,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
        self.device = device
        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, input_model):
        # print(input_model['pixel_values'].shape)
        if len(input_model['pixel_values'].shape) == 5:
            input_model['pixel_values'] = input_model['pixel_values'].squeeze(0)
        print(input_model)
        print(input_model['pixel_values'].shape)
        with torch.no_grad():
            embeddings = self.model.get_image_features(input_model.to(self.device))
        embeddings = f.normalize(embeddings, p=2, dim=1)
        return embeddings

if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenJinaCLIPImageEmbedder()
    count_params(model, verbose=True)