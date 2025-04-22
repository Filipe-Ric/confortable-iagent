from pydantic import BaseModel, Field


class FileRequest(BaseModel):
    bytes_file: bytes