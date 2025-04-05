import os
from zenml import step


@step
def run_spider():
    if not os.path.exists('../../data/raw_dataset.csv'):
        project_dir = "../data/sphinx_value_vision"
        cmd = f"cd {project_dir} && scrapy crawl aqarmap"
        os.system(cmd)


# if __name__ == "__main__":
#     run_spider()
