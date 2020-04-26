import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import io
from tqdm import tqdm

def search(search_term, max_images=50):
	subscription_key = "SEE DISCORD FOR SUBSCRIPTION KEY"
	assert subscription_key is not "SEE DISCORD FOR SUBSCRIPTION KEY"
	search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

	headers = {"Ocp-Apim-Subscription-Key" : subscription_key}

	image_urls = set()

	total = 0
	prevOffset = 0
	nextOffset = 0
	print("Searching...")
	while(total < max_images):
		params  = {"q": search_term,
				"license": "public",
				"count": 16, 
				"offset": nextOffset,  
				"imageType": "photo", 
				#    "maxHeight" : 600,
				#    "maxWidth" : 600
		}
		response = requests.get(search_url, headers=headers, params=params)
		response.raise_for_status()
		search_results = response.json()
		nextOffset = search_results["nextOffset"]
		urls = [img["contentUrl"] for img in search_results["value"][:16]]
		total += len(urls)
		for url in urls:
			image_urls.add(url)
		print(f"{total} found...")
		if nextOffset == prevOffset:
			print("Reached the end of search results!")
			break
	print("Finished search.")
	return image_urls

def download_urls(image_urls, dir="goats", offset=0):
	print("Downloading...")
	failed_saves = 0
	i = offset
	for url in tqdm(list(image_urls)):

		try:
			image_content = requests.get(url).content
			image_file = io.BytesIO(image_content)
			image = Image.open(image_file).convert('RGB')
			image = np.array(image)
			image = cv2.resize(image, (64, 64))
			image = image[:, :, ::-1]
			cv2.imwrite("./data/"+dir+"/{}.png".format(i), image)
			i += 1
			
		except Exception as e:
			# print(f"ERROR - Could not save #{i}  - {e}")
			failed_saves += 1
	print(f"Finished downloading {i} images. {failed_saves} failed to save.")
	return i

# download goat images
# download_urls(search("goat -cow -pig -sheep", 500))

# download non goat images
# offset = download_urls(search("pig -goat", 100), "nongoats")
# offset = download_urls(search("cow -goat", 100), "nongoats", offset)
# offset = download_urls(search("sheep -goat", 200), "nongoats", offset)
# offset = download_urls(search("farm -goat", 50), "nongoats", offset)
# offset = download_urls(search("chicken -goat", 50), "nongoats", offset)
