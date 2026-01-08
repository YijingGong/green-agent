"""Helper functions for paper processing and evaluation."""

import json
import re
from pathlib import Path
from typing import Dict, List

from bert_score import score as bert_score

# Default prompt for participant instructions
DEFAULT_PROMPT = """
You are an expert in dairy science and mathematical modeling. Extract structured information about equations from the provided JSON data and format it according to the template structure.

Extract all equations with their details (variables, parameters, performance metrics, etc.) and paper metadata. Populate all fields according to the template structure.

CRITICAL: Return ONLY valid JSON matching the template structure. No markdown, explanations, or text outside the JSON.
"""


def load_paper_xml(paper_id: str, input_dir: Path) -> str:
    """Load XML content from input directory.
    
    Args:
        paper_id: Paper ID (e.g., "001")
        input_dir: Path to the input directory containing paper XML files
        
    Returns:
        XML string content
        
    Raises:
        FileNotFoundError: If the paper file doesn't exist
    """
    paper_file = input_dir / f"Paper{paper_id}_raw.xml"
    if not paper_file.exists():
        raise FileNotFoundError(f"Paper file not found: {paper_file}")
    return paper_file.read_text(encoding="utf-8")


def load_gold_output(paper_id: str, output_dir: Path) -> str:
    """Load JSON content from output directory.
    
    Args:
        paper_id: Paper ID (e.g., "001")
        output_dir: Path to the output directory containing answer JSON files
        
    Returns:
        JSON string content (for use with evaluate_response())
        
    Raises:
        FileNotFoundError: If the answer file doesn't exist
    """
    answer_file = output_dir / f"Paper{paper_id}_answer.json"
    if not answer_file.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_file}")
    return answer_file.read_text(encoding="utf-8")


def load_template(templates_dir: Path) -> Dict:
    """Load the template JSON structure.
    
    Args:
        templates_dir: Path to the templates directory
        
    Returns:
        Template dictionary structure
        
    Raises:
        FileNotFoundError: If the template file doesn't exist
    """
    template_file = templates_dir / "template_output.json"
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    return json.loads(template_file.read_text(encoding="utf-8"))


def build_participant_payload(xml_content: str, templates_dir: Path) -> str:
    """Build the message payload to send to participant.
    
    Args:
        xml_content: XML content of the paper
        templates_dir: Path to the templates directory
        
    Returns:
        JSON string to send to participant
    """
    template = load_template(templates_dir)
    payload = {
        "xml_content": xml_content,
        "instructions": DEFAULT_PROMPT.strip(),
        "template": template,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_json_string(json_str: str) -> Dict:
    """
    Parse a JSON string, handling potential markdown code blocks or extra text.
    
    Args:
        json_str: String that may contain JSON, possibly wrapped in markdown code blocks
        
    Returns:
        Dict: Parsed JSON data
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    if not json_str or not json_str.strip():
        raise ValueError("Empty JSON string")
    
    text = json_str.strip()
    
    # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(markdown_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    # Try parsing as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try removing # comments (not inside strings)
        lines = []
        for line in text.split('\n'):
            if '#' in line:
                parts = line.split('#', 1)
                if len(parts) == 2:
                    before = parts[0]
                    # Check if # is outside string (even quote count)
                    if (before.count('"') - before.count('\\"')) % 2 == 0:
                        line = parts[0].rstrip()
            lines.append(line)
        return json.loads('\n'.join(lines))


def extract_equations(json_data: Dict) -> List[str]:
    """
    Extract all equation LaTeX strings from JSON structure.
    
    Args:
        json_data: Dictionary containing JSON data with 'equations' key
        
    Returns:
        List of LaTeX equation strings
    """
    equations = []
    for eq in json_data.get('equations', []):
        if isinstance(eq, dict) and isinstance(eq.get('latex'), str):
            equations.append(eq['latex'].strip())
    return equations


def compare_equations_json(json_data1: Dict, json_data2: Dict) -> float:
    """
    Compare two JSON dictionaries and calculate the percentage of extracted equations that exactly match.
    
    Uses the same approach to check through the JSON structure to find equations.
    
    Args:
        json_data1: First JSON data (ground truth or reference)
        json_data2: Second JSON data (prediction or target)
    
    Returns:
        float: Percentage of exact matches (0.0 to 100.0)
        Returns 0.0 if either dict is empty or has no equations.
    """
    if not json_data1 or not json_data2:
        return 0.0
    
    # Extract equations from both JSON structures using the same approach
    equations1 = extract_equations(json_data1)
    equations2 = extract_equations(json_data2)
    
    if len(equations2) == 0:
        return 0.0
    
    # Count exact matches
    # For each equation in json_data2, check if there's an exact match in json_data1
    matched_count = 0
    matched_indices = set()  # Track which equations from json_data1 have been matched
    
    for eq2 in equations2:
        for i, eq1 in enumerate(equations1):
            if i in matched_indices:
                continue  # This equation already matched
            # Exact string match (case-sensitive, whitespace-sensitive)
            if eq2 == eq1:
                matched_count += 1
                matched_indices.add(i)
                break
    
    # Calculate percentage
    percentage = (matched_count / len(equations2)) * 100.0
    return percentage


def calculate_bert_score_json(json_data1: Dict, json_data2: Dict, lang: str = 'en') -> Dict[str, float]:
    """
    Calculate BERTScore for two JSON dictionaries.
    
    Args:
        json_data1: First JSON data (ground truth or reference)
        json_data2: Second JSON data (prediction or target)
        lang: Language code for BERTScore (default: 'en')
    
    Returns:
        Dict with 'precision', 'recall', and 'f1' scores
        
    Raises:
        ValueError: If either JSON data is empty
    """
    if not json_data1 or not json_data2:
        raise ValueError("Both JSON data dictionaries must be non-empty")
    
    # Convert JSON to normalized strings for BERTScore
    json1_str = json.dumps(json_data1, sort_keys=True, ensure_ascii=False)
    json2_str = json.dumps(json_data2, sort_keys=True, ensure_ascii=False)
    
    # Calculate BERTScore
    # Note: bert_score expects (candidates, references) where candidates are predictions
    # So json_data2 is candidate (prediction) and json_data1 is reference (ground truth)
    P, R, F1 = bert_score(
        [json2_str],
        [json1_str],
        lang=lang,
        verbose=False
    )
    
    return {
        'precision': float(P[0]),
        'recall': float(R[0]),
        'f1': float(F1[0])
    }


def evaluate_response(prediction: str, gold_output: str, lang: str = 'en') -> Dict:
    """
    Evaluate a prediction string against a gold output string using both metrics.
    
    This is a convenience function that parses JSON strings and runs both evaluation metrics.
    
    Args:
        prediction: Prediction JSON string (may include markdown formatting)
        gold_output: Gold/reference JSON string (may include markdown formatting)
        lang: Language code for BERTScore (default: 'en')
    
    Returns:
        Dict containing:
        - 'equation_match_percentage': float (0.0-100.0)
        - 'bertscore': dict with 'precision', 'recall', 'f1'
        - 'error': str (if any error occurred during evaluation)
    """
    result = {}
    
    try:
        # Parse JSON strings
        pred_data = parse_json_string(prediction)
        gold_data = parse_json_string(gold_output)
        
        # Calculate equation match percentage
        eq_percentage = compare_equations_json(gold_data, pred_data)
        result['equation_match_percentage'] = eq_percentage
        
        # Calculate BERTScore
        bertscore_result = calculate_bert_score_json(gold_data, pred_data, lang=lang)
        result['bertscore'] = bertscore_result
        
    except Exception as e:
        result['error'] = str(e)
        # Set defaults if error occurred
        if 'equation_match_percentage' not in result:
            result['equation_match_percentage'] = 0.0
        if 'bertscore' not in result:
            result['bertscore'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    return result

