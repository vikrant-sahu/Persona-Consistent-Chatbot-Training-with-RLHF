from setuptools import setup, find_packages

setup(
    name='persona-consistent-chatbot',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A chatbot that maintains persona consistency using reinforcement learning and supervised fine-tuning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/persona-consistent-chatbot',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
        'torch>=1.7.0',
        'transformers>=4.0.0',
        'datasets>=1.0.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'gradio',
        'wandb',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)