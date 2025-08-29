import requests

file_path = r"C:\Users\jacob\OneDrive - UBC\Desktop\Personal Projects\D&D Battlemaps\dnd_battlemaps_lora_dataset_20250812_233504.zip"
with open(file_path, "rb") as f:
    response = requests.put(
        "https://transfer.sh/dnd_battlemaps_lora_dataset.zip", data=f
    )

print("Upload URL:", response.text)
