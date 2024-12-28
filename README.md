# Chatbot Project

This repository contains a simple yet effective chatbot implementation using TensorFlow and NLTK. The chatbot is designed to classify user inputs based on predefined intents and provide appropriate responses.

## Features

- Intent classification using natural language processing.
- Flexible and modular architecture for training and inference.
- Interactive console-based chatbot interface.
- Comprehensive evaluation metrics including confusion matrix and classification report.
- Precision and recall visualization for performance analysis.
- Easy-to-use configuration and model training pipeline.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.10+
- GPU support for TensorFlow (if using a compatible GPU like NVIDIA RTX 3050)
- Required Python libraries (specified in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv chatbot-env
   source chatbot-env/bin/activate  # Linux/MacOS
   chatbot-env\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download necessary NLTK data:
   ```bash
   python
   >>> import nltk
   >>> nltk.download('punkt')
   >>> nltk.download('wordnet')
   >>> nltk.download('omw-1.4')
   >>> exit()
   ```

## Usage

### Training the Chatbot

1. Place your intent data in the `intents.json` file. Follow the provided structure for intents, patterns, and responses.

2. Train the chatbot model:
   ```bash
   python train_chatbot.py
   ```
   This script will preprocess the data, train the model, and save it as `chatbot_model.keras`.

### Running the Chatbot

1. Start the chatbot interface:
   ```bash
   python run_chatbot.py
   ```

2. Type your queries in the console and get responses from the chatbot.

### Evaluation

To evaluate the chatbot's performance, review the classification report and confusion matrix generated during the training process. These can be found in the `results` directory.

## File Structure

```
.
├── intents.json           # Intent definitions
├── train_chatbot.py       # Training script
├── run_chatbot.py         # Chatbot inference script
├── requirements.txt       # Python dependencies
├── results/               # Training results and saved models
├── utils/                 # Utility functions for preprocessing and plotting
└── README.md              # Project documentation
```

## Customization

### Adding New Intents

1. Update the `intents.json` file with new tags, patterns, and responses.
2. Retrain the model by running:
   ```bash
   python train_chatbot.py
   ```

### Adjusting Model Parameters

Modify the hyperparameters in `train_chatbot.py` to suit your dataset and hardware configuration. For example, you can change the number of epochs, batch size, or learning rate.

## Troubleshooting

### Missing NLTK Data

If you encounter errors related to missing NLTK data (e.g., `punkt`, `wordnet`), ensure you have downloaded the necessary packages using the steps in the [Installation](#installation) section.

### TensorFlow GPU Issues

If TensorFlow does not detect your GPU, verify your environment setup, including:

- Compatible TensorFlow and CUDA versions.
- Properly installed NVIDIA drivers.

Refer to the [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu) for more information.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, feature enhancements, or documentation improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [NLTK Library](https://www.nltk.org/)
- [TensorFlow](https://www.tensorflow.org/)
