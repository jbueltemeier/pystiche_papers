from pystiche.image import transforms

__all__ = [
    "preprocessor",
    "postprocessor",
]

def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()