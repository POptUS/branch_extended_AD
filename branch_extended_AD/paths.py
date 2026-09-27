def path_key(path):
    if not isinstance(path, list):
        return path
    return tuple((node.name, node.choices[0]) for node in path)


def paths_equal(path1, path2):
    if not isinstance(path1, list) or not isinstance(path2, list):
        return path1 == path2
    if len(path1) != len(path2):
        return False
    return all(
        node1.name == node2.name
        and len(node1.choices) == 1
        and len(node2.choices) == 1
        and node1.choices[0] == node2.choices[0]
        for node1, node2 in zip(path1, path2)
    )


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def paths_any_in(needles, haystack):
    haystack_keys = {path_key(path) for path in haystack}
    return any(path_key(path) in haystack_keys for path in needles)


def paths_all_in(needles, haystack):
    haystack_keys = {path_key(path) for path in haystack}
    return all(path_key(path) in haystack_keys for path in needles)
