from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DashScope
    dashscope_api_key: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/rag.db"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_parse_topic: str = "document.parse.requests"
    kafka_parse_group: str = "parse-workers"

    # File Storage
    upload_dir: str = "./data/uploads"
    parsed_dir: str = "./data/parsed"

    # Service
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Embedding dimensions
    text_embedding_dim: int = 1024
    image_embedding_dim: int = 512

    # Retrieval
    top_k_text: int = 5
    top_k_images: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
