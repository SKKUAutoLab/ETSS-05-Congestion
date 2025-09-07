from setuptools import setup, find_packages

setup(name="rankbench", version="0.1.0", description="Benchmarking representations for ranking", long_description=open("README.md").read(), long_description_content_type="text/markdown",
      author="Ankit Sonthalia", author_email="ankitsonthalia24@gmail.com", url="https://github.com/aktsonthalia/rankanything", package_dir={"": "src"}, packages=find_packages(where="src"),
      install_requires=["numpy>=1.18.0", "scipy>=1.4.0"], python_requires=">=3.7", classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent"], entry_points={"console_scripts": ["your_command=your_package.module:function"]}, include_package_data=True, zip_safe=False)