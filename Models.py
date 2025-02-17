from pydantic import BaseModel


class ComentarioDTO(BaseModel):
    texto: str