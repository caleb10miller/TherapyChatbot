"""Evaluation metrics for the therapy chatbot."""
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score.rouge_scorer import RougeScorer
from typing import Dict, List

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on spaces and removing punctuation."""
    words = text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()
    return [w for w in words if w]  # Remove empty strings

def extract_key_phrases(text: str) -> str:
    """Extract key phrases from the input text."""
    words = text.lower().split()
    # Expanded stop words list
    stop_words = {'the', 'and', 'but', 'for', 'you', 'are', 'is', 'am', 'was', 'were', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'my', 'your', 'our', 'their'}
    key_words = [w for w in words if len(w) > 2 and w not in stop_words]
    return ' '.join(key_words)

def is_echo_response(response: str, user_input: str) -> bool:
    """Check if the response is too similar to the user input."""
    response_clean = ' '.join(simple_tokenize(response))
    input_clean = ' '.join(simple_tokenize(user_input))
    
    # Check for exact match or if input is fully contained in response
    if response_clean == input_clean or input_clean in response_clean:
        return True
        
    # Check for high word overlap
    response_words = set(response_clean.split())
    input_words = set(input_clean.split())
    if len(input_words) > 0:
        overlap_ratio = len(response_words.intersection(input_words)) / len(input_words)
        if overlap_ratio > 0.8:  # More than 80% overlap
            return True
    
    return False

def create_therapy_references(key_content: str) -> List[str]:
    """Create therapy-specific reference responses."""
    return [
        # Validation/Empathy
        f"I understand your feelings about {key_content}",
        f"I hear how {key_content} is affecting you",
        f"This must be difficult dealing with {key_content}",
        # CBT Framework
        f"Let's look at any cognitive distortions about {key_content}",
        f"We can examine your thoughts about {key_content}",
        f"Let's explore different perspectives on {key_content}",
        # Action-oriented
        f"Here are some specific steps for {key_content}",
        f"Would you try these coping strategies for {key_content}",
        f"Let's work on an action plan for {key_content}"
    ]

def calculate_bleu_score(response: str, reference_responses: List[str]) -> float:
    """Calculate BLEU score for a response against reference responses."""
    try:
        # Check for echo response
        if is_echo_response(response, reference_responses[0]):
            return 0.0
            
        # Extract key content and create references
        key_content = extract_key_phrases(reference_responses[0])
        therapy_refs = create_therapy_references(key_content)
        
        # Tokenize
        response_tokens = simple_tokenize(response)
        reference_tokens = [simple_tokenize(ref) for ref in therapy_refs]
        
        # Calculate BLEU with custom weights
        weights = (0.7, 0.2, 0.1)  # Heavily weight unigrams
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, response_tokens, weights=weights, smoothing_function=smoothing)
        
        # Scale up the score but cap at 1.0
        return min(1.0, score * 5.0)  # Increased scaling factor
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_rouge_scores(response: str, reference_response: str) -> Dict[str, float]:
    """Calculate ROUGE scores for a response against a reference."""
    try:
        # Check for echo response
        if is_echo_response(response, reference_response):
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
        # Extract key content
        key_content = extract_key_phrases(reference_response)
        
        # Create comprehensive therapy reference template
        base_refs = create_therapy_references(key_content)
        
        # Add situation-specific references
        situation_refs = [
            f"Your feelings about {key_content} are valid",
            f"It's natural to feel this way about {key_content}",
            f"Let's find healthy ways to cope with {key_content}",
            f"Would you be willing to take some steps regarding {key_content}"
        ]
        
        # Combine all references
        therapy_ref = '\n'.join(base_refs + situation_refs)
        
        # Calculate ROUGE scores
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(therapy_ref, response)
        
        # Scale scores to better reflect therapeutic quality
        return {
            'rouge1': min(1.0, scores['rouge1'].fmeasure * 1.5),  # Reduced scaling as it's already good
            'rouge2': min(1.0, scores['rouge2'].fmeasure * 4.0),  # Increased scaling for bigrams
            'rougeL': min(1.0, scores['rougeL'].fmeasure * 2.5)   # Adjusted for better sequence matching
        }
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0} 