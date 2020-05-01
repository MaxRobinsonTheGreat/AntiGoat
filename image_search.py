import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from tqdm import tqdm
import pickle

def bing_search(search_term, max_images=50):
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
			image_file = BytesIO(image_content)
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

# **********************************************
### BING SEARCH - use these sparingly. I am only allowed 1000 total api calls this month
# search and save goat urls
# goats = bing_search("goat", 400)
# goats.update(bing_search("white goat", 200))
# goats.update(bing_search("brown goat", 100))
# goats.update(bing_search("black goat", 100))
# goats.update(bing_search("billy goat", 100))
# with open('./data/goat_urls.data', 'wb') as filehandle:
#     pickle.dump(list(goats), filehandle)

# search and save non goat urls
# nongoats = search("pig -goat", 100)
# nongoats.update(search("cow -goat", 100))
# nongoats.update(search("sheep -goat", 500))
# nongoats.update(search("farm -goat", 100))
# nongoats.update(search("chicken -goat", 100))
# with open('./data/nongoat_urls.data', 'wb') as filehandle:
#     pickle.dump(list(nongoats), filehandle)


# **********************************************
### IMAGE DOWNLOADS - use as much as you want
# download goat images
# with open('./data/goat_urls.data', 'rb') as filehandle:
# 	goat_urls = pickle.load(filehandle)
# 	download_urls(goat_urls)

# download nongoat images
# with open('./data/nongoat_urls.data', 'rb') as filehandle:
#     goat_urls = pickle.load(filehandle)
#     download_urls(goat_urls, "nongoats")


# **********************************************
### UPDATE SEARCH - use sparingly
# old_urls = pickle.load(open('./data/nongoat_urls.data', 'rb'))
# old_urls = set(old_urls)
# old_urls.update(bing_search("white sheep", 100))
# old_urls.update(bing_search("lamb animal", 200))
# old_urls.update(bing_search("brown sheep", 200))
# print(len(old_urls))
# pickle.dump(list(old_urls), open('./data/nongoat_urls.data', 'wb'))
