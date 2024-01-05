import git


def get_head_sha() -> str:
    """Позволяет получить последний хэш коммита текущего репозитория."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha[:8]  # short form of commit hash
