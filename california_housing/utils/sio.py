from pathlib import Path
from typing import Any

from skops import io as sio
from skops.io.exceptions import UntrustedTypesFoundException


def safe_skops_load(artifact_path: str | Path, add_types: list[str]) -> Any:
    """
    Позволяет безопасно загрузить сериализованные через skops артефакты (модели, процессинги и тд).
    :param artifact_path: путь до артефакта.
    :param add_types: дополнительные типы, разрешенные для класса. Необходимо использовать, если
    мы сериализуем кастомные классы поверх sklearn.
    :returns: загруженный артефакт.
    :raises: исключение, если объект содержит недоверенные типы.
    """
    found_types = sio.get_untrusted_types(file=artifact_path)
    if all(
        type_.startswith(("numpy", "scipy", "sklearn", *add_types))
        for type_ in found_types
    ):
        artifact = sio.load(artifact_path, trusted=True)
        return artifact
    else:
        raise UntrustedTypesFoundException(found_types)
