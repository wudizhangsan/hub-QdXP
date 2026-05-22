from pymilvus import MilvusClient, connections, Collection, FieldSchema, CollectionSchema, DataType

from app.config import settings

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "images"

_text_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="page_start", dtype=DataType.INT64),
        FieldSchema(name="page_end", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.text_embedding_dim),
    ],
    description="Text chunks with BGE embeddings",
)

_image_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page_num", dtype=DataType.INT64),
        FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.image_embedding_dim),
    ],
    description="Images with CLIP embeddings",
)

_collections = {
    TEXT_COLLECTION: _text_schema,
    IMAGE_COLLECTION: _image_schema,
}


def connect_milvus():
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    return MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")


def init_milvus(client: MilvusClient):
    """Create collections if they don't exist."""
    for name, schema in _collections.items():
        if not client.has_collection(name):
            client.create_collection(collection_name=name, schema=schema)
            _create_index(client, name)


def _create_index(client: MilvusClient, collection_name: str):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(collection_name, index_params)


def get_client() -> MilvusClient:
    return connect_milvus()
