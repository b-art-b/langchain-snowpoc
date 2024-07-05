from setuptools import setup, find_packages

setup(
    name="langchain-snowpoc",
    version="0.0.1",
    packages=find_packages(),
    description="This is a PoC for Snowflake integration with Langchain",
    author="Bart Wrobel",
    author_email="124384994+b-art-b@users.noreply.github.com",
    url="https://github.com/b-art-b/langchain-snowpoc",
    install_requires=[
        "langchain==0.1.12",
        "langchain-core==0.1.32",
        "pyarrow==15.0.2",
        "snowflake-connector-python==3.7.1",
        "snowflake-ml-python==1.3.0",
        "snowflake-snowpark-python==1.13.0",
    ],
)
