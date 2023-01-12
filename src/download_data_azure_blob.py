
import time
from azure.storage.blob import BlobServiceClient

blob_service_client = BlobServiceClient.from_connection_string(
    ""
)

container_client = blob_service_client.get_container_client("")

polling_interval = 1
data_download_dir = '/home/works/Pictures/data'
while True:
    # Get a list of all the blobs in the container
    blob_list = container_client.list_blobs()

    # Find the latest blob
    latest_blob = None
    latest_blob_time = None
    for blob in blob_list:
        if (latest_blob_time is None) or (blob.creation_time > latest_blob_time):
            latest_blob = blob
            latest_blob_time = blob.creation_time

    if latest_blob:
        save_file_name = data_download_dir + "/" + str(latest_blob.name)
        with open(save_file_name, "wb") as local_file:
            container_client.download_blob(latest_blob.name).readinto(local_file)
        container_client.delete_blob(latest_blob)
    time.sleep(polling_interval)
