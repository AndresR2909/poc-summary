from __future__ import annotations

import logging
import os

from azure.storage.blob import BlobServiceClient
from azure.storage.blob import StorageStreamDownloader
from dotenv import load_dotenv
load_dotenv()

#logging.basicConfig(level=logging.ERROR)

connectionString = os.environ.get('BLOBSTORAGE_CONNECTION_STRING')



class BlobStorage:
    def __init__(self) -> None:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connectionString)
            self.container_client = blob_service_client

        except Exception as e:
            logging.error(f"Error connecting to blob storage: {e}")

    def list_files_from_folder_container(self, container_name, folder_name:str):
        try:
            blob_list = [{'filename': elemento.name, 'lastupdate': elemento.last_modified}
                 for elemento in self.container_client.get_container_client(container_name).list_blobs(name_starts_with=folder_name)]
            return blob_list
        except Exception as e:
            print(f"error: {str(e)}, message: error al listar archivos en la carpeta")
            return None

    def download_file_from_blob(self, container_name, blob_name:str)->StorageStreamDownloader:
        try:

            file = self.container_client.get_container_client(container_name).download_blob(blob_name)

            return file

        except Exception as e:
            logging.error(f"error: {str(e)}, message: error al descargar el archivo")
            return None

    def upload_blob(self, container_name, blob_name,file_data):
        try:

            blob_client = self.container_client.get_container_client(container_name).get_blob_client(blob_name)
            blob_client.upload_blob(file_data, overwrite=True)

            return {'filename': blob_name, 'message': 'Archivo cargado exitosamente en Blob Storage'}

        except Exception as e:
            return {'error': str(e), 'message': 'error al cargar el archivo'}

    def move_blob(self, source_container_name,source_blob_name, destination_container_name, destination_blob_name):
        try:
            source_blob_client = self.container_client.get_container_client(source_container_name).get_blob_client(source_blob_name)
            destination_blob_client = self.container_client.get_container_client(destination_container_name).get_blob_client(destination_blob_name)

            if source_blob_client.exists():
                # Download the source blob
                source_blob_data = source_blob_client.download_blob().readall()

                # Upload the data to the destination blob
                destination_blob_client.upload_blob(source_blob_data, overwrite=True)

                # Delete the source blob
                source_blob_client.delete_blob()

                return {'message': f"Archivo movido de '{source_blob_name}' a '{destination_blob_name}' exitosamente."}
            else:
                return {'error': 'El archivo de origen no existe.'}

        except Exception as e:
            return {'error': str(e), 'message': 'error al mover el archivo'}
