from setuptools import setup, find_packages

setup(
    name="googlemodel-samrat",
    version="0.1.5",
    packages=find_packages(include=["googlemodel_samrat*"]),
    package_data={"googlemodel_samrat": ["py.typed"]},
    install_requires=[
        "langchain-google-genai>=4.0.0",
        "python-dotenv>=1.0.0",
        "google-api-core>=2.15.0",
        "langchain-core>=0.1.5",
        "pydantic>=2.0",
    ],
)