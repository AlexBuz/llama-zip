from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="llama-zip",
    version="0.4.0",
    description="LLM-powered compression tool",
    author="Alexander Buzanis",
    packages=find_packages(),
    py_modules=["llama_zip"],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llama-zip=llama_zip:main",
        ],
    },
)
