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

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git
- Virtual environment tool (venv or conda)
- At least 4GB RAM
- Stable internet connection

### System Requirements

- Operating System: Windows 10+, macOS 10.15+, or Linux
- Disk Space: At least 1GB free space
- Memory: 4GB RAM minimum, 8GB recommended
- Network: Stable internet connection for API calls

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

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-3.5-turbo  # or your fine-tuned model name
```

### Command Options

The chatbot supports the following commands:
- `quit`, `exit`, or `bye` - End the current session

The chatbot automatically:
- Evaluates each response using multiple metrics
- Calculates BLEU and ROUGE scores
- Saves conversation history to logs

## Using the Chatbot

The main chatbot interface (`main.py`) provides an interactive way to engage with the therapy chatbot:

1. **Starting a Conversation**
   ```bash
   python src/main.py
   ```
   This launches the interactive chat interface where you can:
   - Have a conversation with the chatbot
   - Get real-time response evaluations
   - View BLEU and ROUGE scores
   - Save conversation history

2. **Response Features**
   - Real-time response generation
   - Automatic evaluation of responses
   - Conversation context maintenance
   - Support for both base and fine-tuned models

3. **Example Usage**
   ```
   > I've been feeling anxious lately
   [Chatbot responds with therapeutic support]
   
   Response Evaluation:
   Empathy: 0.85
   Clarity: 0.92
   CBT Technique: 0.78
   Supportiveness: 0.88
   Response Quality: 0.90
   BLEU score: 0.45
   ROUGE-1: 0.65
   ROUGE-2: 0.45
   ROUGE-L: 0.60
   ```

## Response Evaluation

The chatbot includes a sophisticated evaluation system that scores responses using multiple metrics:

### Automatic Evaluation Metrics

1. **BLEU Score (0-1)**
   - Measures response similarity to therapeutic reference responses
   - Uses custom weights (70% unigrams, 20% bigrams, 10% trigrams)
   - Includes echo response detection
   - Scaled to better reflect therapeutic quality

2. **ROUGE Scores (0-1)**
   - **ROUGE-1**: Unigram overlap (scaled by 1.5x)
   - **ROUGE-2**: Bigram overlap (scaled by 4.0x)
   - **ROUGE-L**: Longest common subsequence (scaled by 2.5x)
   - Includes echo response detection
   - Uses comprehensive therapy reference templates

### Echo Response Detection

The system automatically detects and penalizes echo responses:
- Exact matches with user input
- High word overlap (>80%)
- Input fully contained in response

### Reference Response Generation

The system generates therapy-specific reference responses based on:
- Validation and empathy phrases
- CBT framework responses
- Action-oriented suggestions
- Situation-specific references

## Model Evaluation and Comparison

The project includes a comprehensive model evaluation system that compares different models across multiple dimensions:

### Supported Models

1. **Base Models**
   - GPT-3.5 Turbo 0125
   - GPT-4
   - GPT-4 Turbo Preview
   - GPT-4 0125 Preview
   - GPT-4o Mini

2. **Fine-tuned Models**
   - Fine-tuned GPT-3.5 Turbo 0125

### Evaluation Process

1. **Test Setup**
   - Uses standardized test prompts from `data/test_prompts.json`
   - Tests each model with and without system prompts
   - Consistent evaluation criteria across all models

2. **System Prompt**
   ```python
   You are a supportive therapy chatbot specializing in Cognitive Behavioral Therapy (CBT).
   Your responses should be empathetic, clear, and incorporate CBT techniques when appropriate.
   Focus on validating feelings, challenging cognitive distortions, and providing practical support.
   ```

3. **Evaluation Metrics**
   - Empathy
   - Clarity
   - CBT Technique
   - Supportiveness
   - Response Quality
   - BLEU Score
   - ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)

4. **Echo Response Detection**
   - Checks for exact matches
   - Measures similarity ratio
   - Applies threshold-based filtering

### Results Analysis

1. **Output Files**
   - Detailed results CSV with all metrics per response
   - Summary statistics CSV with mean scores per model
   - Timestamped files for tracking improvements

2. **Analysis Features**
   - Per-model performance metrics
   - Impact of system prompts
   - Category-specific analysis
   - Echo response rates

### Running the Comparison

```bash
python src/evaluation/model_comparison.py
```

This will:
1. Load test prompts
2. Test each model configuration
3. Generate evaluation metrics
4. Save detailed and summary results
5. Display summary statistics

### Example Output

```
Model Comparison Results:
Model                                Empathy  Clarity  CBT Tech  Support  Quality
gpt-3.5-turbo-0125                   0.744    0.800    0.583    0.745    0.790
gpt-3.5-turbo-0125 + prompt          0.712    0.788    0.567    0.733    0.778
gpt-4                                0.540    0.723    0.160    0.612    0.721
gpt-4 + prompt                       0.605    0.725    0.220    0.645    0.735
...
```

## Project Structure

```
.
├── data/               # Training and validation data
│   ├── finetune/      # Processed data for fine-tuning
│   ├── train_data.csv # Training dataset
│   ├── val_data.csv   # Validation dataset
│   ├── test_data.csv  # Test dataset
│   └── test_prompts.json # Test prompts for evaluation
├── models/            # Model implementations
│   ├── gpt_model.py   # GPT model implementation
│   ├── base_model.py  # Base model interface
│   └── __init__.py    # Model package initialization
├── src/              # Source code
│   ├── evaluation/   # Response evaluation system
│   │   ├── metrics.py           # Evaluation metrics implementation
│   │   ├── model_comparison.py  # Model comparison and analysis
│   │   └── __init__.py         # Evaluation package initialization
│   ├── inference/    # Chatbot interface
│   ├── training/     # Fine-tuning scripts
│   ├── main.py       # Main application
│   └── __init__.py   # Source package initialization
├── notebooks/        # Jupyter notebooks for analysis
│   └── data_collection.ipynb  # Data collection and analysis
├── logs/            # Conversation logs
│   ├── model_comparison/  # Model comparison results
│   ├── gpt-4/           # GPT-4 model logs
│   ├── gpt-3.5-turbo/   # GPT-3.5 model logs
│   └── ft:gpt-3.5-turbo-0125:personal::BWtfMrrB/  # Fine-tuned model logs
├── requirements.txt # Project dependencies
├── .env            # Environment configuration
├── .gitignore     # Git ignore rules
├── LICENSE        # MIT License
└── README.md      # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 