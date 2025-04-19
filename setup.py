from setuptools import setup, find_packages

setup(
    name="lunar-lander-ppo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium[box2d]==0.29.1",
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
        "PyYAML>=6.0",
        "moviepy>=1.0.0",
        "pillow>=8.0.0",
        "scikit-learn>=1.0.0",
    ],
    author="Kevin Mok",
    author_email="mokkevi1@msu.edu",
    description="A PPO implementation for the LunarLander-v2 environment",
    keywords="reinforcement-learning, ppo, lunar-lander, gymnasium",
    url="https://github.com/kevmok/lunar-lander-ppo",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
) 