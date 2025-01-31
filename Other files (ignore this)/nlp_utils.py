import re
from bs4 import BeautifulSoup

from torch.utils.data import Dataset

def clean_text(text):
    # Step 1: Remove inline JavaScript/CSS
    text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL)
    
    # Step 2: Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Step 3: Remove remaining HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')  # Extract text and separate with spaces
    
    # Step 4: General regex to remove any encodings like =XX (two hexadecimal digits)
    text = re.sub(r'=[0-9A-Fa-f]{2}', ' ', text)

    # Step 5: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove standalone single characters
    text = re.sub(r'\b\w\b', '', text)

    # Remove prefixed 'b'
    text = text.lstrip('b')

    # Remove any extra spaces again, just to be sure
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    return text


# Create a custom Dataset class to return a dict for each batch
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

