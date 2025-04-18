from __future__ import annotations

import logging
import os
from typing import Dict
from typing import List
from typing import Optional

from azure.storage.filedatalake import FileSystemClient

logging.basicConfig(level=logging.ERROR)

connection_string = os.environ.get('DATALAKE_CONNECTION_STRING')


class DataLakeStorage:
    def __init__(self, zone: str) -> None:
        try:
            self.dl_client = FileSystemClient.from_connection_string(
                connection_string, zone,
            )
        except Exception as e:
            logging.error(f"Error connecting to blob storage: {e}")

    def list_files_from_path(
        self, directory_name: str,
    ) -> Optional[List[Dict[str, str]]]:
        try:
            paths = self.dl_client.get_paths(path=directory_name)
            file_list = [
                {'filename': f.name, 'lastupdate': f.last_modified.isoformat()}
                for f in paths
            ]
            return file_list
        except Exception as e:
            logging.error(f"Error listing files from path: {e}")
            return None
