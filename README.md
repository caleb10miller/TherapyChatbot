# Therapy Chatbot

A conversational AI chatbot designed to provide therapeutic support and guidance using GPT models with a focus on Cognitive Behavioral Therapy (CBT) techniques.

## Description

This project implements an AI-powered chatbot that engages in meaningful conversations and provides therapeutic support using CBT techniques. The chatbot uses GPT models (base or fine-tuned) and includes a robust evaluation system to ensure response quality.

## Features

- Natural language conversation using GPT models
- Empathetic responses with CBT techniques
- Comprehensive response evaluation system:
  - Empathy scoring
  - Clarity assessment
  - CBT technique application
  - Supportiveness evaluation
  - Response quality metrics
- Conversation history management
- Fine-tuning support for custom training
- BLEU and ROUGE score tracking

## Getting Started

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   MODEL_NAME=gpt-3.5-turbo  # or your fine-tuned model
   ```
5. Run the chatbot:
   ```bash
   python src/main.py
   ```

## Response Evaluation

The chatbot includes a comprehensive evaluation system that scores responses on five key dimensions:

1. **Empathy (0-1)**
   - 0.0-0.29: No emotional validation
   - 0.3-0.49: Basic acknowledgment
   - 0.5-0.69: Good validation
   - 0.7-0.89: Excellent validation
   - 0.9-1.0: Perfect validation

2. **Clarity (0-1)**
   - 0.0-0.29: Unclear response
   - 0.3-0.49: Basic explanation
   - 0.5-0.69: Clear explanation
   - 0.7-0.89: Very clear
   - 0.9-1.0: Perfect clarity

3. **CBT Technique (0-1)**
   - 0.0-0.29: No technique used
   - 0.3-0.49: Basic mention
   - 0.5-0.69: Good explanation
   - 0.7-0.89: Excellent application
   - 0.9-1.0: Perfect application

4. **Supportiveness (0-1)**
   - 0.0-0.29: No practical advice
   - 0.3-0.49: Basic support
   - 0.5-0.69: Good advice
   - 0.7-0.89: Excellent steps
   - 0.9-1.0: Perfect support

5. **Response Quality (0-1)**
   - 0.0-0.29: Poor structure/length
   - 0.3-0.49: Basic structure
   - 0.5-0.69: Good structure
   - 0.7-0.89: Excellent structure
   - 0.9-1.0: Perfect structure

Automatic penalties:
- Responses under 15 words: All scores reduced by 50%

## Fine-tuning

To fine-tune the model on your own data:

1. Prepare your training data in CSV format with 'prompt' and 'utterance' columns
2. Run the data preparation script:
   ```bash
   python src/training/prepare_finetune.py
   ```
3. Start the fine-tuning process:
   ```bash
   python src/training/finetune_gpt.py
   ```

After fine-tuning is complete, update your `.env` file with the fine-tuned model name:
```
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=ft:gpt-3.5-turbo:your-org:your-model-id
```

## Project Structure

```
.
├── data/               # Training and validation data
│   ├── finetune/      # Processed data for fine-tuning
│   ├── train_data.csv # Training dataset
│   ├── val_data.csv   # Validation dataset
│   └── test_data.csv  # Test dataset
├── models/            # Model implementations
│   ├── gpt_model.py   # GPT model implementation
│   ├── base_model.py  # Base model interface
│   └── __init__.py    # Model package initialization
├── src/              # Source code
│   ├── evaluation/   # Response evaluation system
│   ├── inference/    # Chatbot interface
│   ├── training/     # Fine-tuning scripts
│   ├── main.py       # Main application
│   └── __init__.py   # Source package initialization
├── notebooks/        # Jupyter notebooks for analysis
├── logs/            # Conversation logs
├── tests/           # Unit tests
├── requirements.txt # Project dependencies
├── LICENSE         # MIT License
└── README.md       # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 