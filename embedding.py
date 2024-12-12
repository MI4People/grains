import os
import json
import boto3
import openai
from PyPDF2 import PdfReader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

s3 = boto3.client('s3')
bucket_name = 'grains-files'
prefix = 'house-keeping/'

openai.api_key = os.environ['OPENAI_API_KEY']

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

processed_file = 'processed_documents.json'
if os.path.exists(processed_file):
	with open(processed_file, 'r') as f:
		processed_docs = set(json.load(f))
else:
	processed_docs = set()

vector_store = {}

def create_vector_store():
	print("Vector store initialized.")

create_vector_store()

def chunk_text_semantically(text, max_length=500):
	sentences = sent_tokenize(text)
	chunks = []
	current_chunk = []

	for sentence in sentences:
		if len(" ".join(current_chunk + [sentence])) <= max_length:
			current_chunk.append(sentence)
		else:
			chunks.append(" ".join(current_chunk))
			current_chunk = [sentence]

	if current_chunk:
		chunks.append(" ".join(current_chunk))

	return chunks

def process_pdf(file_path):
	reader = PdfReader(file_path)
	text = ""
	images = []

	for page in reader.pages:
		if page.extract_text():
			text += page.extract_text() + "\n"

		if '/XObject' in page.get('/Resources', {}):
			xObject = page['/Resources']['/XObject'].get_object()
			for obj in xObject:
				if xObject[obj]['/Subtype'] == '/Image':
					size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
					data = xObject[obj].get_data()
					mode = "RGB" if xObject[obj]['/ColorSpace'] == '/DeviceRGB' else "L"
					img = Image.frombytes(mode, size, data)
					images.append(img)

	return text, images

response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
if 'Contents' in response:
	for obj in response['Contents']:
		key = obj['Key']
		if key not in processed_docs and key.endswith('.pdf'):
			print(f"Processing: {key}")

			local_file = key.replace('/', '_')
			s3.download_file(bucket_name, key, local_file)

			text, images = process_pdf(local_file)

			chunks = chunk_text_semantically(text)
			text_embeddings = [
				openai.Embedding.create(input=chunk, model="text-embedding-3-large")["data"][0]["embedding"]
				for chunk in chunks
			]

			image_embeddings = []
			for image in images:
				inputs = clip_processor(text=None, images=image, return_tensors="pt", padding=True)
				outputs = clip_model.get_image_features(**inputs)
				image_embeddings.append(outputs.detach().numpy().tolist())

			vector_store[key] = {"text_embeddings": text_embeddings, "image_embeddings": image_embeddings}

			processed_docs.add(key)

			os.remove(local_file)

with open(processed_file, 'w') as f:
	json.dump(list(processed_docs), f)

print("Processing complete. Processed documents updated.")
