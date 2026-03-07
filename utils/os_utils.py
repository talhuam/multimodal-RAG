import os


def get_sorted_md_files(dir_name):
    """
    获取按文件名排序后的文件路径，由于段落可能跨页
    Args:
        dir_name (str): 文件夹路径
    Returns:
        文件列表
    """
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith('.md')]

    def sort_key(file_path):
        file_without_ext, _ = os.path.splitext(file_path)
        return int(file_without_ext.split("_")[-1])
    return sorted(files, key=sort_key, reverse=False)


# if __name__ == '__main__':
#     print(get_sorted_md_files("../data/Attention Is All You Need"))