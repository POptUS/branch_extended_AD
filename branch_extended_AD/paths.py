def path_key(path):
    """Return a hashable key for one resolved path."""
    if not isinstance(path, list):
        raise TypeError("path must be a list of resolved trace nodes")
    return tuple((node.name, node.choices[0]) for node in path)


def paths_equal(path1, path2):
    """Return whether two resolved paths contain the same choices."""
    return path_key(path1) == path_key(path2)


def unique_paths(paths):
    """Remove duplicate paths while keeping their original order."""
    seen = set()
    result = []
    for path in paths:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def paths_any_in(needles, haystack):
    """Return whether any requested path occurs in another collection."""
    haystack_keys = {path_key(path) for path in haystack}
    return any(path_key(path) in haystack_keys for path in needles)


def paths_all_in(needles, haystack):
    """Return whether every requested path occurs in another collection."""
    haystack_keys = {path_key(path) for path in haystack}
    return all(path_key(path) in haystack_keys for path in needles)
