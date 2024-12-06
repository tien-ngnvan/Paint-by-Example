from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
import torch.nn.functional as f
device = "cpu"
model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)

# image_urls = [
#     'https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg',
#     # 'https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg',
#     # 'https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg',
#     # 'https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg',
#     # 'https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg',
#     # 'https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg',
# ]
preprocessor = AutoImageProcessor.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
image = "figure/result_1.png"
image = Image.open(image).convert('RGB')
print(image.size)
pixelvals = preprocessor(image).to(device)
print(pixelvals)
print(pixelvals['pixel_values'].shape)
embeddings = model.get_image_features(pixelvals)
embeddings = f.normalize(embeddings, p=2, dim=1)
print(embeddings.shape)