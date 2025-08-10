import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS
# ============================================================================

# Set your paths here
IMAGE_FOLDER = "/kaggle/input/images"           # Path to your images folder
MODEL_PATH = "/kaggle/input/weights2/pytorch/default/1/coco_weights.pt"   # Path to your ClipCap model file
OUTPUT_FILE = "/kaggle/working/captions.xlsx"     # Where to save the Excel file
PREFIX_LENGTH = 10                         # Should match your model configuration
MAX_CAPTION_LENGTH = 50                    # Maximum length of generated captions

# ============================================================================

# Handle transformers import
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    print("✓ Transformers loaded successfully")
except ImportError as e:
    print(f"❌ Error importing transformers: {e}")
    print("Installing transformers...")
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Handle CLIP import with multiple fallbacks
CLIP_MODEL = None
CLIP_PREPROCESS = None

def install_and_load_clip():
    global CLIP_MODEL, CLIP_PREPROCESS
    
    # Try original CLIP first
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        CLIP_MODEL = model
        CLIP_PREPROCESS = preprocess
        print("✓ Loaded original OpenAI CLIP")
        return True
    except ImportError:
        print("Original CLIP not available, installing...")
        
        try:
            # Install original CLIP
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            CLIP_MODEL = model
            CLIP_PREPROCESS = preprocess
            print("✓ Installed and loaded original OpenAI CLIP")
            return True
        except:
            print("Failed to install original CLIP, trying open_clip...")
    
    # Try open_clip as fallback
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        CLIP_MODEL = model
        CLIP_PREPROCESS = preprocess
        print("✓ Loaded open_clip")
        return True
    except ImportError:
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            CLIP_MODEL = model
            CLIP_PREPROCESS = preprocess
            print("✓ Installed and loaded open_clip")
            return True
        except:
            print("❌ Failed to install open_clip")
    
    print("❌ No CLIP implementation could be loaded!")
    return False

class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length=10, prefix_size=512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, 
                                self.gpt_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size, device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse=True):
        return self.clip_project.parameters()

    def train(self, mode=True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def encode_image(image_path, device):
    """Encode image using available CLIP model"""
    global CLIP_MODEL, CLIP_PREPROCESS
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = CLIP_PREPROCESS(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = CLIP_MODEL.encode_image(image_tensor)
        return features.float()

def generate_caption(model, tokenizer, image_path, device, max_length=50):
    """Generate caption for a single image using greedy decoding"""
    model.eval()
    
    # Encode image
    clip_features = encode_image(image_path, device)
    
    with torch.no_grad():
        prefix_embed = model.clip_project(clip_features).view(1, model.prefix_length, -1)
        
        # Start with empty sequence
        generated = torch.tensor([]).long().to(device).unsqueeze(0)
        
        for i in range(max_length):
            # Get embeddings for generated tokens
            if generated.shape[1] > 0:
                embeddings = model.gpt.transformer.wte(generated)
                combined_embeds = torch.cat([prefix_embed, embeddings], dim=1)
            else:
                combined_embeds = prefix_embed
            
            # Get next token prediction
            outputs = model.gpt(inputs_embeds=combined_embeds)
            logits = outputs.logits[0, -1, :]
            
            # Sample next token (greedy)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Check for end token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated = torch.cat([generated, next_token], dim=1)
        
        # Decode to text
        if generated.shape[1] > 0:
            return tokenizer.decode(generated[0], skip_special_tokens=True)
        else:
            return "Unable to generate caption"

def load_model(model_path, device, prefix_length=10):
    """Load the trained ClipCap model"""
    try:
        model = ClipCaptionPrefix(prefix_length=prefix_length)
        
        # Try loading with strict=False to handle potential key mismatches
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def run_captioning():
    """Main function to run the captioning process"""
    
    print("🚀 Starting ClipCap Image Captioning...")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Error: Image folder '{IMAGE_FOLDER}' does not exist")
        print("Please update the IMAGE_FOLDER path in the configuration section")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' does not exist")
        print("Please update the MODEL_PATH in the configuration section")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Install and load CLIP
    print("📦 Loading CLIP model...")
    if not install_and_load_clip():
        return
    
    CLIP_MODEL.to(device)
    CLIP_MODEL.eval()
    
    # Load GPT-2 tokenizer
    print("📝 Loading GPT-2 tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load ClipCap model
    print("🤖 Loading ClipCap model...")
    model = load_model(MODEL_PATH, device, PREFIX_LENGTH)
    if model is None:
        return
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_folder = Path(IMAGE_FOLDER)
    image_files = [f for f in image_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No image files found in {IMAGE_FOLDER}")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"📸 Found {len(image_files)} images")
    print("🔄 Generating captions...")
    print("-" * 60)
    
    # Process images
    results = []
    successful = 0
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing"), 1):
        try:
            caption = generate_caption(model, tokenizer, image_file, device, MAX_CAPTION_LENGTH)
            results.append({
                'image_name': image_file.name,
                'image_path': str(image_file),
                'caption': caption.strip()
            })
            print(f"✓ [{i}/{len(image_files)}] {image_file.name}")
            print(f"   📝 {caption.strip()}")
            print()
            successful += 1
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            results.append({
                'image_name': image_file.name,
                'image_path': str(image_file),
                'caption': error_msg
            })
            print(f"❌ [{i}/{len(image_files)}] Error processing {image_file.name}: {str(e)}")
    
    # Save to Excel
    print("💾 Saving results to Excel...")
    try:
        # Install openpyxl if not available
        try:
            import openpyxl
        except ImportError:
            print("Installing openpyxl...")
            
        
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"✅ Results saved to {OUTPUT_FILE}")
        print(f"📊 Successfully processed {successful}/{len(image_files)} images")
        
        # Display first few results
        print("\n📋 Preview of results:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")
        print("\n📋 Results (displayed instead):")
        for result in results:
            print(f"{result['image_name']}: {result['caption']}")

# Instructions for use
def show_instructions():
    print("🔧 SETUP INSTRUCTIONS:")
    print("=" * 50)
    print("1. Update the paths in the CONFIGURATION SECTION at the top:")
    print(f"   - IMAGE_FOLDER = '{IMAGE_FOLDER}'")
    print(f"   - MODEL_PATH = '{MODEL_PATH}'")
    print(f"   - OUTPUT_FILE = '{OUTPUT_FILE}'")
    print()
    print("2. Run the captioning process:")
    print("   run_captioning()")
    print()
    print("📁 Make sure your:")
    print("   - Images are in JPG, PNG, BMP, or TIFF format")
    print("   - ClipCap model file is a .pt file")
    print("   - Paths use forward slashes (/)")

# Run instructions by default
show_instructions()
run_captioning()
