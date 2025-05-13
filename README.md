# Therapy Chatbot

A conversational AI chatbot designed to provide therapeutic support and guidance using GPT-3.5 Turbo.

## Description

This project implements an AI-powered chatbot that engages in meaningful conversations and provides therapeutic support using Cognitive Behavioral Therapy (CBT) techniques. The chatbot is fine-tuned on GPT-3.5 Turbo to provide empathetic and supportive responses while maintaining appropriate boundaries.

## Features

- Natural language conversation using GPT-3.5 Turbo
- Empathetic responses with CBT techniques
- Response quality evaluation
- Conversation history management
- Fine-tuning support for custom training

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
   MODEL_NAME=gpt-3.5-turbo
   ```
5. Run the chatbot:
   ```bash
   python src/main.py
   ```

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

## Project Structure

```
therapy-chatbot/
├── data/               # Training and validation data
│   └── finetune/      # Processed data for fine-tuning
├── models/            # Model implementations
│   ├── gpt_model.py   # GPT-3.5 Turbo implementation
│   └── base_model.py  # Base model interface
├── src/              # Source code
│   ├── training/     # Fine-tuning scripts
│   ├── inference/    # Chatbot interface
│   └── main.py       # Main application
└── tests/            # Unit tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 