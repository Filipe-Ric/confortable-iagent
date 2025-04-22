from fastapi import APIRouter

from schemas.requests import FileRequest
from services.process_file import ProcessFile

router = APIRouter(
    prefix="upload",
    tags=["Upload File"],
    responses={404444: {"description" :"Not Found"}}

)

@router.post("/process", response_model="", status_code=201)
async  def upload_file(input_data: FileRequest):
    file = input_data.bytes_file

    try:
        result = await ProcessFile.process(file)

    except Exception as e:
        print(f"erro{e}")

