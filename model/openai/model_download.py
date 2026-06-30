import requests
import os
url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"

# url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
save_path = "./CLIP-ViT-L-14/ViT-L-14.pt"  

os.makedirs(os.path.dirname(save_path), exist_ok=True)

file_size = 0
if os.path.exists(save_path):
    file_size = os.path.getsize(save_path)
    print(f"Found some downloaded files, {file_size} bytes have been downloaded, will continue downloading...")

headers = {"Range": f"bytes={file_size}-"} if file_size > 0 else {}

try:
    with requests.get(url, headers=headers, stream=True, timeout=30) as response:
        response.raise_for_status() 
        
        total_size = int(response.headers.get("content-length", 0)) + file_size
        
        with open(save_path, "ab") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    file_size += len(chunk)
                    if total_size > 0:
                        progress = (file_size / total_size) * 100
                        print(f"\n{progress:.2f}% ({file_size}/{total_size})", end="")
    
    print("\n finished !")

except KeyboardInterrupt:
    print("\n error !!")
except Exception as e:
    print(f"\n error: {str(e)}")