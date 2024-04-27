def cluster_overlapping_strings(string_list):
    clusters = []
    while string_list:
        current_string = string_list.pop(0)
        current_cluster = [current_string]
        overlapping_strings = set(current_string.split())
        i = 0
        while i < len(string_list):
            if any(word in overlapping_strings for word in string_list[i].split()):
                current_cluster.append(string_list.pop(i))
            else:
                i += 1
        clusters.append(current_cluster)
    return clusters