
from .vit import ViTEmbeddingModel
from .cnn import CNNEmbeddingModel

class ModelFactory:
    """
    Factory class for creating embedding models.
    """
    @staticmethod
    def create(model_type: str, **kwargs) -> object:
        """
        Creates an embedding model based on the specified type.
        Args:
            model_type (str): Type of the model to create (e.g., 'cnn', 'vit').
            **kwargs: Additional keyword arguments for the model constructor.
        Returns:
            An instance of the specified embedding model.
        Raises:
            ValueError: If the model type is unknown.
        """
        if model_type == "cnn":
            return CNNEmbeddingModel(**kwargs)
        elif model_type == "vit":
            return ViTEmbeddingModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
